const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Web stubs for native-only packages
const WEB_ALIASES = {
  'react-native-vision-camera': path.resolve(__dirname, 'src/mocks/react-native-vision-camera.web.ts'),
  'ffmpeg-kit-react-native': path.resolve(__dirname, 'src/mocks/ffmpeg-kit-react-native.web.ts'),
  'react-native-share': path.resolve(__dirname, 'src/mocks/react-native-share.web.ts'),
};

config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (platform === 'web' && WEB_ALIASES[moduleName] != null) {
    return { filePath: WEB_ALIASES[moduleName], type: 'sourceFile' };
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
