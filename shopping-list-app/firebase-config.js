// ===== Firebase Configuration =====
//
// To enable syncing between devices:
// 1. Go to https://console.firebase.google.com/
// 2. Create a new project (free tier is plenty)
// 3. Enable Firestore Database (start in test mode)
// 4. Go to Project Settings > General > Your apps > Add web app
// 5. Copy your config values below
//
// Without Firebase, the app works fully offline with local storage.

window.FIREBASE_CONFIG = null; // Set to null to disable sync

// To enable sync, uncomment and fill in your Firebase config:
/*
window.FIREBASE_CONFIG = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};
*/

// Load Firebase SDK dynamically only if config is set
(function() {
  if (!window.FIREBASE_CONFIG) return;

  const scripts = [
    'https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js',
    'https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore-compat.js',
  ];

  let loaded = 0;
  scripts.forEach(src => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => {
      loaded++;
      if (loaded === scripts.length) {
        // Fire a custom event so app.js knows Firebase is ready
        document.dispatchEvent(new Event('firebase-ready'));
      }
    };
    document.head.appendChild(script);
  });
})();
