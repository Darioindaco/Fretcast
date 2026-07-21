// Web stub for ffmpeg-kit-react-native
export const FFmpegKit = {
  execute: async (_cmd: string) => ({ getReturnCode: async () => ({ isValueSuccess: () => false }) }),
  executeAsync: async (_cmd: string, _cb?: unknown) => null,
};
export const ReturnCode = { isSuccess: (_code: unknown) => false };
