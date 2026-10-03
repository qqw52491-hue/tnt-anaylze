// mac_stt: 按住右 Option 录音 → /tmp/tnt_voice.wav(16kHz mono) → stdout 行协议
// 重建(本机 swiftc 与 SDK 不匹配,故用 clang/ObjC):
//   clang -O2 -fobjc-arc -framework Foundation -framework AVFoundation \
//         -framework CoreGraphics -Wl,-sectcreate,__TEXT,__info_plist,stt/Info.plist \
//         mac_stt.m -o mac_stt && codesign -s - --force mac_stt
// (codesign 必须重跑:把内嵌 Info.plist 绑进签名,否则 TCC 读不到用途说明)
//   READY            初始化完成,开始监听按键
//   REC:START        按下右 Option,开始录音
//   REC:END          松开
//   WAV:<path>       录音文件已就绪(每次覆盖同一路径)
//   ERR:<msg>        错误/权限问题
//   ./mac_stt tap    仅做按键回显测试(不开麦)
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>

static const char *kWavPath = "/tmp/tnt_voice.wav";
// 默认监听键码集合:56=左Shift 60=右Shift,按住任何一个都算。
// 可用 TNT_VOICE_KEY="键码,逗号分隔" 覆盖(./mac_stt keys 可查键码)
static const char *kDefaultPttKeys = "56,60";

// 解析 "58,59,61" → CGKeyCode 数组
static int parseKeys(const char *s, CGKeyCode *out, int maxn) {
    int n = 0;
    char *buf = strdup(s);
    for (char *tok = strtok(buf, ","); tok && n < maxn; tok = strtok(NULL, ","))
        out[n++] = (CGKeyCode)atoi(tok);
    free(buf);
    return n;
}

static BOOL ensureMicAuth(void) {
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio]) {
        case AVAuthorizationStatusAuthorized: return YES;
        case AVAuthorizationStatusNotDetermined: {
            __block BOOL ok = NO;
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                                     completionHandler:^(BOOL g) {
                                         ok = g;
                                         dispatch_semaphore_signal(sem);
                                     }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 240LL * NSEC_PER_SEC));
            if (!ok) printf("ERR:mic_denied(系统设置>隐私>麦克风 里允许)\n");
            return ok;
        }
        default:
            printf("ERR:mic_denied(系统设置>隐私>麦克风 里允许)\n");
            return NO;
    }
}

@interface HoldRec : NSObject {
    AVAudioRecorder *_rec;
    BOOL _recording;
    NSDate *_recStart;
}
- (void)keyDown;
- (void)keyUp;
@end

@implementation HoldRec
- (void)keyDown {
    if (_recording) return;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kWavPath]];
    NSDictionary *settings = @{
        AVFormatIDKey : @(kAudioFormatLinearPCM),
        AVSampleRateKey : @16000.0,
        AVNumberOfChannelsKey : @1,
        AVLinearPCMBitDepthKey : @16,
        AVLinearPCMIsFloatKey : @NO,
        AVLinearPCMIsBigEndianKey : @NO,
    };
    NSError *err = nil;
    _rec = [[AVAudioRecorder alloc] initWithURL:url settings:settings error:&err];
    if (!_rec) {
        printf("ERR:recorder_init %s\n", err.localizedDescription.UTF8String);
        return;
    }
    [_rec prepareToRecord];
    if (![_rec record]) {
        printf("ERR:record_start(没插麦克风?Mac mini 无内置麦,插 USB麦/耳机/AirPods)\n");
        _rec = nil;
        return;
    }
    _recording = YES;
    _recStart = [NSDate date];
    printf("REC:START\n");
}

- (void)keyUp {
    if (!_recording) return;
    _recording = NO;
    printf("REC:END\n");
    NSTimeInterval dur = [[NSDate date] timeIntervalSinceDate:_recStart];
    [_rec stop];
    _rec = nil;
    if (dur < 0.3) {
        printf("WAV:(too_short)\n");
    } else {
        printf("WAV:%s\n", kWavPath);
    }
}
@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        // 调试: ./mac_stt keys —— 打印所有按键按下/松开的原始键码(找按键用)
        if (argc >= 2 && strcmp(argv[1], "keys") == 0) {
            printf("READY 按任意键看键码,Ctrl+C 退出\n");
            BOOL *state = calloc(256, sizeof(BOOL));
            [NSTimer scheduledTimerWithTimeInterval:0.02
                                            repeats:YES
                                              block:^(NSTimer *t) {
                                                  for (int k = 0; k < 256; k++) {
                                                      BOOL d = CGEventSourceKeyState(
                                                          kCGEventSourceStateCombinedSessionState,
                                                          (CGKeyCode)k);
                                                      if (d != state[k]) {
                                                          printf("KEY %d %s\n", k,
                                                                 d ? "DOWN" : "UP");
                                                          state[k] = d;
                                                      }
                                                  }
                                              }];
            [[NSRunLoop mainRunLoop] run];
            return 0;
        }

        // 常听模式: ./mac_stt stream —— 麦克风持续录音,stdout 输出
        // 原始 s16 PCM(16kHz 单声道)。日志/错误走 stderr。
        if (argc >= 2 && strcmp(argv[1], "stream") == 0) {
            if (!ensureMicAuth()) {
                fprintf(stderr, "ERR:mic_denied\n");
                return 2;
            }
            AVAudioEngine *engine = [[AVAudioEngine alloc] init];
            AVAudioInputNode *input = engine.inputNode;
            AVAudioFormat *dst = [[AVAudioFormat alloc]
                initWithCommonFormat:AVAudioPCMFormatInt16
                          sampleRate:16000.0
                            channels:1
                         interleaved:YES];
            __block AVAudioConverter *conv = nil;
            __block AVAudioFormat *srcFmt = nil;
            [input installTapOnBus:0
                        bufferSize:1024
                            format:nil
                             block:^(AVAudioPCMBuffer *buf, AVAudioTime *t) {
                                 if (!conv || srcFmt != buf.format) {
                                     srcFmt = buf.format;
                                     conv = [[AVAudioConverter alloc]
                                         initFromFormat:buf.format
                                              toFormat:dst];
                                     if (!conv) {
                                         fprintf(stderr, "ERR:no_conv\n");
                                         return;
                                     }
                                 }
                                 AVAudioPCMBuffer *out = [[AVAudioPCMBuffer alloc]
                                     initWithPCMFormat:dst
                                     frameCapacity:(AVAudioFrameCount)(buf.frameLength * 16000.0 / buf.format.sampleRate) + 64];
                                 __block BOOL fed = NO;
                                 [conv convertToBuffer:out
                                                 error:nil
                                    withInputFromBlock:^(AVAudioPacketCount n,
                                                         AVAudioConverterInputStatus *st) {
                                        if (fed) {
                                            *st = AVAudioConverterInputStatus_NoDataNow;
                                            return (AVAudioBuffer *)nil;
                                        }
                                        fed = YES;
                                        *st = AVAudioConverterInputStatus_HaveData;
                                        return (AVAudioBuffer *)buf;
                                    }];
                                 if (out.frameLength > 0)
                                     fwrite(out.int16ChannelData[0], 2,
                                            out.frameLength, stdout);
                             }];
            NSError *err = nil;
            [engine startAndReturnError:&err];
            if (err) {
                fprintf(stderr, "ERR:engine %s\n", err.localizedDescription.UTF8String);
                return 3;
            }
            fprintf(stderr, "READY stream\n");
            [[NSRunLoop mainRunLoop] run];
            return 0;
        }

        // 调试: ./mac_stt rec <秒> —— 录 N 秒写 wav 后退出
        if (argc >= 3 && strcmp(argv[1], "rec") == 0) {
            if (!ensureMicAuth()) return 2;
            HoldRec *hr = [[HoldRec alloc] init];
            [hr keyDown];
            [NSThread sleepForTimeInterval:atof(argv[2])];
            [hr keyUp];
            [NSThread sleepForTimeInterval:0.2];
            return 0;
        }

        if (argc >= 2 && strcmp(argv[1], "tap") == 0) {
            // 调试:回显按键按下/松开,不碰麦克风; ./mac_stt tap <键码>
            CGKeyCode k = (argc >= 3) ? (CGKeyCode)atoi(argv[2]) : 59;
            printf("READY key=%d\n", (int)k);
            __block BOOL last = NO;
            [NSTimer scheduledTimerWithTimeInterval:0.03
                                            repeats:YES
                                              block:^(NSTimer *t) {
                                                  BOOL d = CGEventSourceKeyState(
                                                      kCGEventSourceStateCombinedSessionState,
                                                      k);
                                                  if (d != last) {
                                                      printf("%s\n", d ? "DOWN" : "UP");
                                                      last = d;
                                                  }
                                              }];
            [[NSRunLoop mainRunLoop] run];
            return 0;
        }

        if (!ensureMicAuth()) return 2;
        HoldRec *hr = [[HoldRec alloc] init];

        // 按键可配: ./mac_stt <键码或逗号列表> 或 TNT_VOICE_KEY 环境变量
        const char *keySpec = (argc >= 2) ? argv[1]
                                        : (getenv("TNT_VOICE_KEY") ?: kDefaultPttKeys);
        CGKeyCode keys[16];
        int nkeys = parseKeys(keySpec, keys, 16);
        CGKeyCode *keysPtr = malloc(sizeof(CGKeyCode) * nkeys);
        memcpy(keysPtr, keys, sizeof(CGKeyCode) * nkeys);
        printf("READY ptt=%s\n", keySpec);

        __block int nkeysB = nkeys;
        [NSTimer scheduledTimerWithTimeInterval:0.03
                                        repeats:YES
                                          block:^(NSTimer *t) {
                                              BOOL down = NO;
                                              for (int i = 0; i < nkeysB; i++) {
                                                  if (CGEventSourceKeyState(
                                                          kCGEventSourceStateCombinedSessionState,
                                                          keysPtr[i])) {
                                                      down = YES;
                                                      break;
                                                  }
                                              }
                                              if (down)
                                                  [hr keyDown];
                                              else
                                                  [hr keyUp];
                                          }];

        // 父进程退出(stdin EOF)则自杀,避免孤儿进程一直占着麦
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            char buf[64];
            while (read(STDIN_FILENO, buf, sizeof(buf)) > 0) {
            }
            exit(0);
        });

        [[NSRunLoop mainRunLoop] run];
    }
    return 0;
}
