# The Shopping List

A shared shopping list PWA with voice input, store tabs, and aisle memory that learns where items are in your supermarkets.

## Features

- **Voice input** - Tap the mic and say "tinned tomatoes - Aldi" to add items
- **Store tabs** - Items automatically grouped by store (Aldi, Sainsbury's, etc.)
- **Real-time sync** - Both you and your partner see the same list via Firebase
- **Duplicate detection** - Catches duplicates even with different wording ("tinned tomatoes" vs "canned tomatoes")
- **Aisle memory** - When you tick an item off, it asks which aisle you found it in. Next time you add that item, it shows the aisle location
- **Smart ordering** - Items sorted by aisle so you can shop in order
- **Split aisle support** - Large stores (like big Sainsbury's) with walkways through aisles get A/B half labels
- **Works offline** - PWA caches everything locally, syncs when back online

## Quick Start (No Sync)

The app works immediately without any setup - just open `index.html` in a browser. Items are saved in your browser's local storage.

To serve it properly (needed for voice input and PWA install):

```bash
cd shopping-list-app
python3 -m http.server 8000
# Open http://localhost:8000 on your phone
```

Or use any static file server (Node: `npx serve`, VS Code Live Server, etc.)

## Setting Up Sync (Firebase)

To sync between two phones:

### 1. Create a Firebase Project (Free)
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" - give it any name
3. Disable Google Analytics (not needed) and create

### 2. Enable Firestore
1. In your Firebase project, go to **Build > Firestore Database**
2. Click "Create database"
3. Choose **Start in test mode** (fine for personal use)
4. Select a region close to you and create

### 3. Get Your Config
1. Go to **Project Settings** (gear icon) > **General**
2. Scroll to "Your apps" > click the web icon (`</>`)
3. Register app with any name
4. Copy the `firebaseConfig` object

### 4. Add Config to the App
Edit `firebase-config.js` and uncomment/fill in your config:

```js
window.FIREBASE_CONFIG = {
  apiKey: "AIzaSy...",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

### 5. Deploy & Share
The easiest way to host it for free:

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login and init
firebase login
firebase init hosting
# Select your project, set public directory to "." , say yes to single-page app

# Deploy
firebase deploy
```

This gives you a URL like `https://your-project.web.app` - share it with your partner.

### 6. Link Your Lists
1. Open the app on your phone
2. Tap the settings icon (top right)
3. Tap "New" to generate a list ID
4. Tap "Copy List ID to Share" and send it to your partner
5. Your partner opens the app, goes to Settings, pastes the ID, and taps "Apply"
6. You're synced!

## Aisle System

When you tick an item as bought, the app asks where you found it:

- **Aisle number** (1, 2, 3...)
- **Side** (Left / Right as you walk in)
- **Position** (Start / Middle / End of the aisle)

For **large stores with split aisles** (where a walkway cuts through the middle):
- Go to Settings and toggle "Split aisles" for that store
- The picker adds a **Front half (A) / Back half (B)** option

This creates labels like:
- `Aisle 3 - left side - middle` (standard)
- `Aisle 5B - right side - start` (split aisle store)

The app remembers aisle locations per item per store, so next time you add "milk" to "Aldi", it automatically shows where to find it.

## How Voice Input Works

The app uses the Web Speech API (built into Chrome/Safari). Say things like:
- "milk" (adds to current tab's store)
- "bread Aldi" (adds to Aldi)
- "gluten free pasta Sainsbury's" (adds to Sainsbury's)
- "tinned tomatoes Lidl" (creates Lidl tab if it doesn't exist)

It recognises store names even with slight variations ("Sainsbury's", "Sainsburys", "sainos").

## Browser Support

- **Chrome/Edge** (Android & Desktop) - Full support including voice
- **Safari** (iOS) - Full support including voice
- **Firefox** - Everything except voice input (no Web Speech API)
