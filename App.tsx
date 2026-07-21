import { Platform } from 'react-native';

// Reanimated must be first import on native — skip on web (worklets not needed for web preview)
if (Platform.OS !== 'web') {
  require('react-native-reanimated');
}

import { GestureHandlerRootView } from 'react-native-gesture-handler';
import Navigation from './src/navigation';

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <Navigation />
    </GestureHandlerRootView>
  );
}
