// stt_rec: SenseVoice(sherpa-onnx) 常驻识别服务——模型只加载一次
//   模式1: stt_rec <model.int8.onnx> <tokens.txt>
//          stdin 每行一个 wav 路径 → stdout: TEXT:<文本> / ERR:<原因>
//   模式2: stt_rec vad <model.int8.onnx> <tokens.txt> <silero_vad.onnx>
//          stdin 连续原始 s16 PCM(16kHz 单声道) → VAD 自动切段识别 → TEXT:<文本>
//   启动成功后打 READY
// 重建:
//   clang -O2 -I stt/sherpa-onnx-v1.12.40-osx-arm64-shared/include \
//         stt/stt_rec.c -L stt/sherpa-onnx-v1.12.40-osx-arm64-shared/lib \
//         -lsherpa-onnx-c-api \
//         -Wl,-rpath,@executable_path/stt/sherpa-onnx-v1.12.40-osx-arm64-shared/lib \
//         -o stt_rec
#include <stdio.h>
#include <string.h>
#include "sherpa-onnx/c-api/c-api.h"

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        printf("ERR:usage stt_rec [vad] <model.onnx> <tokens.txt> [vad.onnx]\n");
        return 2;
    }

    // vad 模式:参数右移一位
    int vad_mode = strcmp(argv[1], "vad") == 0;
    int arg0 = vad_mode ? 2 : 1;

    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.model_config.sense_voice.model = argv[arg0];
    config.model_config.sense_voice.language = "zh";
    config.model_config.sense_voice.use_itn = 1;
    config.model_config.tokens = argv[arg0 + 1];
    config.model_config.num_threads = 4;
    config.decoding_method = "greedy_search";

    const SherpaOnnxOfflineRecognizer *rec = SherpaOnnxCreateOfflineRecognizer(&config);
    if (!rec) {
        printf("ERR:model_load_failed\n");
        return 2;
    }

    if (vad_mode) {
        // ---- VAD 常听模式:stdin = 连续 s16 PCM @16kHz ----
        if (argc <= arg0 + 2) {
            printf("ERR:vad_mode_needs_vad_onnx\n");
            return 2;
        }
        SherpaOnnxVadModelConfig vc;
        memset(&vc, 0, sizeof(vc));
        vc.silero_vad.model = argv[arg0 + 2];
        vc.silero_vad.threshold = 0.5f;
        vc.silero_vad.min_silence_duration = 0.4f; // 停顿0.4s即切句
        vc.silero_vad.min_speech_duration = 0.15f;
        vc.silero_vad.max_speech_duration = 15.0f;
        vc.silero_vad.window_size = 512;
        vc.sample_rate = 16000;
        vc.num_threads = 1;
        const SherpaOnnxVoiceActivityDetector *vad =
            SherpaOnnxCreateVoiceActivityDetector(&vc, 30.0f);
        if (!vad) {
            printf("ERR:vad_load_failed\n");
            return 2;
        }
        printf("READY\n");
        int16_t pcm[512];
        float f32[512];
        while (fread(pcm, sizeof(int16_t), 512, stdin) == 512) {
            for (int i = 0; i < 512; i++) f32[i] = pcm[i] / 32768.0f;
            SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, f32, 512);
            while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
                const SherpaOnnxSpeechSegment *seg =
                    SherpaOnnxVoiceActivityDetectorFront(vad);
                const SherpaOnnxOfflineStream *s =
                    SherpaOnnxCreateOfflineStream(rec);
                SherpaOnnxAcceptWaveformOffline(s, 16000, seg->samples, seg->n);
                SherpaOnnxDecodeOfflineStream(rec, s);
                const SherpaOnnxOfflineRecognizerResult *r =
                    SherpaOnnxGetOfflineStreamResult(s);
                printf("TEXT:%s\n", (r && r->text) ? r->text : "");
                if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
                SherpaOnnxDestroyOfflineStream(s);
                SherpaOnnxDestroySpeechSegment(seg);
                SherpaOnnxVoiceActivityDetectorPop(vad);
            }
        }
        // EOF:冲掉 VAD 里没收尾的语音段
        SherpaOnnxVoiceActivityDetectorFlush(vad);
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment *seg =
                SherpaOnnxVoiceActivityDetectorFront(vad);
            const SherpaOnnxOfflineStream *s = SherpaOnnxCreateOfflineStream(rec);
            SherpaOnnxAcceptWaveformOffline(s, 16000, seg->samples, seg->n);
            SherpaOnnxDecodeOfflineStream(rec, s);
            const SherpaOnnxOfflineRecognizerResult *r =
                SherpaOnnxGetOfflineStreamResult(s);
            printf("TEXT:%s\n", (r && r->text) ? r->text : "");
            if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
            SherpaOnnxDestroyOfflineStream(s);
            SherpaOnnxDestroySpeechSegment(seg);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
        return 0;
    }

    printf("READY\n");

    char line[2048];
    while (fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        const SherpaOnnxWave *wave = SherpaOnnxReadWave(line);
        if (!wave) {
            printf("ERR:bad_wav\n");
            continue;
        }
        const SherpaOnnxOfflineStream *s = SherpaOnnxCreateOfflineStream(rec);
        SherpaOnnxAcceptWaveformOffline(s, wave->sample_rate, wave->samples,
                                        wave->num_samples);
        SherpaOnnxDecodeOfflineStream(rec, s);
        const SherpaOnnxOfflineRecognizerResult *r =
            SherpaOnnxGetOfflineStreamResult(s);
        printf("TEXT:%s\n", (r && r->text) ? r->text : "");
        if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
        SherpaOnnxDestroyOfflineStream(s);
        SherpaOnnxFreeWave(wave);
    }
    return 0;
}
