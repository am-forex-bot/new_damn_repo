# The Shopping List - Setup Guide

A shared shopping list app for you and your wife. Voice input, store tabs, and it learns where items are in each supermarket.

---

## What You Need

- A laptop/computer (for the one-time setup)
- Both your Android phones
- A Google account (for Firebase - free)

---

## Step 1: Create a Firebase Project (on your laptop)

This is what syncs the list between your phones. It's free.

1. Open **https://console.firebase.google.com/** in your browser
2. Click **"Create a project"**
3. Name it anything (e.g. "shopping-list") and click Continue
4. Turn **off** Google Analytics (you don't need it) and click **Create project**
5. Wait for it to finish, then click **Continue**

## Step 2: Enable the Database (on your laptop)

1. On the project home page, click the **"Build a backend"** card
2. Select **Cloud Firestore** and click **Create database**
3. Select **"Start in test mode"** and click Next
4. Pick the location closest to you (e.g. `europe-west2` for UK) and click **Enable**

## Step 3: Get Your Firebase Config (on your laptop)

1. Click the **Home** icon (top left) to go back to the project home page
2. Click **"+ Add app"** (under the project name at the top)
3. Click the **web icon** (`</>`)
4. Enter any nickname (e.g. "shopping list") and click **Register app**
5. You'll see a code block with `firebaseConfig = { ... }` - keep this page open, you need these values

## Step 4: Add Your Config to the App (on your laptop)

1. Open the file `shopping-list-app/firebase-config.js` in a text editor
2. Replace the contents with your values from Step 3. It should look like this:

```js
window.FIREBASE_CONFIG = {
  apiKey: "AIzaSyB1234...",
  authDomain: "shopping-list-abc.firebaseapp.com",
  projectId: "shopping-list-abc",
  storageBucket: "shopping-list-abc.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123def456"
};

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
        document.dispatchEvent(new Event('firebase-ready'));
      }
    };
    document.head.appendChild(script);
  });
})();
```

3. Save the file

## Step 5: Deploy the App (on your laptop)

You need Node.js installed. If you don't have it: **https://nodejs.org/** (download the LTS version).

Open a terminal and run:

```bash
# Install Firebase CLI (one-time)
npm install -g firebase-tools

# Login to Firebase
firebase login

# Go to the app folder
cd shopping-list-app

# Set up hosting
firebase init hosting
```

When it asks:
- **Select a project** → pick the one you created in Step 1
- **Public directory** → type `.` (just a dot)
- **Single-page app** → type `y`
- **Overwrite index.html** → type `N`

Then deploy:

```bash
firebase deploy
```

It will give you a URL like `https://shopping-list-abc.web.app` — **this is your app!**

## Step 6: Install on Your Phone (on your Android phone)

1. Open **Chrome** on your Android phone
2. Go to the URL from Step 5 (e.g. `https://shopping-list-abc.web.app`)
3. Chrome will show a banner saying **"Add to Home screen"** — tap it
   - If no banner appears: tap the **three dots menu** (top right) → **"Add to Home screen"** → **"Add"**
4. The app icon now appears on your home screen like a normal app

## Step 7: Create and Share Your List ID (on your phone)

This is how both phones see the same list:

1. Open the app
2. Tap the **⚙ settings icon** (top right)
3. Tap **"New"** to generate a list ID
4. Tap **"Apply"** — you should see "Connected & syncing" in green
5. Tap **"Share via WhatsApp"** — this opens WhatsApp with a message containing the list ID and instructions for your wife

## Step 8: Your Wife's Phone (on her Android phone)

1. She opens the WhatsApp message you sent
2. She taps the link to open the app in Chrome
3. She adds it to her home screen (same as Step 6)
4. She opens the app, taps **⚙ settings**
5. She pastes the list ID from the WhatsApp message
6. She taps **"Apply"**
7. Done — you're both synced!

---

## How to Use It

### Adding Items
- **Voice**: Tap the mic button and say something like:
  - "tinned plum tomatoes Aldi"
  - "GF bread Sainsbury's"
  - "salted peanuts"  (adds to whichever store tab you're on)
- **Type**: Type in the text box, e.g. `milk - Aldi` and tap +

### Ticking Items Off
- Tap the circle next to an item to mark it as bought
- It will ask **"Where did you find this?"** — tap the aisle, side, and position
- Tap **Skip** if you don't want to record the aisle
- Next time you add that item to that store, the aisle info shows automatically

### Aisle System
- **Aisle number**: The aisle number in the store
- **Side**: Left or Right as you walk into the aisle
- **Position**: Start (near entrance), Middle, or End (far end)
- **Split aisles** (for big Sainsbury's): Go to Settings → toggle "Split aisles" for that store. This adds Front half (A) / Back half (B) — A is the entrance side of the walkway, B is the far side

### Managing Stores
- Say a new store name when adding an item and it auto-creates the tab
- Or tap the **+** button next to the tabs
- Remove stores in Settings

### Clearing the List
- After a shop, go to Settings → **"Remove all ticked items"** to clear everything you've bought
