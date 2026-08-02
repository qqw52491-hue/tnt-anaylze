#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: %s <image_path>\n", argv[0]);
            return 1;
        }

        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        NSData *data = [NSData dataWithContentsOfURL:url];
        if (!data) {
            printf("Error: Could not read file at %s\n", argv[1]);
            return 1;
        }

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithData:data options:@{}];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.recognitionLanguages = @[@"zh-Hans", @"zh-Hant", @"en-US"];
        request.usesLanguageCorrection = YES;

        NSError *error = nil;
        [handler performRequests:@[request] error:&error];
        if (error) {
            printf("OCR Error: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        for (VNRecognizedTextObservation *obs in request.results) {
            VNRecognizedText *text = [[obs topCandidates:1] firstObject];
            if (text) {
                printf("%s\n", [text.string UTF8String]);
            }
        }
    }
    return 0;
}
