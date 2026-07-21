module.exports = function (api) {
  api.cache(true);

  const plugins = [];

  // react-native-reanimated/plugin MUST be last — but skip on web
  // (worklets plugin crashes Metro web bundler)
  if (process.env.EXPO_PLATFORM !== 'web' && process.env.NODE_ENV !== 'test') {
    plugins.push('react-native-reanimated/plugin');
  }

  return {
    presets: ['babel-preset-expo'],
    plugins,
  };
};
