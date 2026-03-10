# The Shopping List - Setup Guide

A shared shopping list app for you and your wife. Voice input, store tabs, and it learns where items are in each supermarket.

---

## What You Need

- **Your laptop** — for the one-time Firebase setup and deploy
- **Both your Android phones** — to use the app
- **A Google account** — for Firebase (completely free)
- **Node.js on your laptop** — needed to deploy. If you don't have it, download the LTS version from https://nodejs.org/ and install it before starting

---

## Step 1: Create a Firebase Project (on your laptop)

Firebase is what syncs the list between your phones. It's free.

1. Open your browser and go to **https://console.firebase.google.com/**
2. Sign in with your Google account if prompted
3. Click **"Create a project"** (or "Add project")
4. **Project name:** Type anything, e.g. `shoppinglist`
5. Click **Continue**
6. It will ask about Google Analytics — **toggle it OFF** (you don't need it)
7. Click **Create project**
8. Wait 10-15 seconds for it to finish, then click **Continue**

You should now see your project's home page. It'll be full of Gemini AI suggestion cards — **ignore all of those**.

---

## Step 2: Create the Database (on your laptop)

This is the database that stores your shopping list and syncs it between phones.

### Getting to Firestore

The sidebar on the left only shows a few small icons by default. Look for a small **`>` arrow at the very bottom-left** of the page and click it to expand the full sidebar menu. Then:

1. In the expanded sidebar, look for **"Firestore Database"** and click it
2. Click the **"Create database"** button

**Can't find it?** Go directly to this URL in your browser:
```
https://console.firebase.google.com/project/YOUR-PROJECT-ID/firestore
```
(Replace `YOUR-PROJECT-ID` with your project name — check your browser's address bar to find it, it'll be something like `shoppinglist-a1b2c`)

### The Create Database Wizard

You'll see a 3-step wizard:

#### Screen 1 — "Select edition"
- You'll see two options: **Standard edition** and **Enterprise edition**
- **Standard edition** should already be selected (it's highlighted in blue) — that's the right one
- Click **Next**

#### Screen 2 — "Database ID & location"
- **Database ID:** Leave it as `(default)` — don't change this
- **Location:** Click the dropdown and pick one close to you. For the UK, choose **`europe-west2 (London)`**
- Click **Next**

#### Screen 3 — "Configure"
This is about security rules. You'll see two options:

- **"Start in test mode"** — pick this one. It lets the app read and write data freely (fine for a personal app between you and your wife)
- "Start in locked mode" — don't pick this

Click **Create**

Wait a few seconds. You'll then see the Firestore database page (it'll be empty — that's fine, the app creates everything it needs automatically).

---

## Step 3: Register a Web App & Get Your Config (on your laptop)

Now you need to tell Firebase this is a web app, and get the connection details.

1. Click the **Home icon** in the top-left of the sidebar (the little house) to go back to the project home page
2. Under the project name at the top, you'll see a **"+ Add app"** button — click it
3. You'll see icons for different platforms (iOS, Android, Web, Unity, Flutter). Click the **Web icon** — it looks like `</>`
4. **App nickname:** Type anything, e.g. `shopping list`
5. **Tick the box** that says "Also set up Firebase Hosting for this app" — this is important, it's how you'll publish the app
6. Click **Register app**
7. You'll now see a code block on screen. It contains something like this:

```
const firebaseConfig = {
  apiKey: "AIzaSyB...",
  authDomain: "shoppinglist-a1b2c.firebaseapp.com",
  projectId: "shoppinglist-a1b2c",
  storageBucket: "shoppinglist-a1b2c.firebasestorage.app",
  messagingSenderId: "123456789012",
  appId: "1:123456789012:web:abc123def456"
};
```

8. **Keep this page open** or copy these values somewhere — you need them for the next step
9. Click **Continue to console**

---

## Step 4: Add Your Config to the App (on your laptop)

1. Open the file `shopping-list-app/firebase-config.js` in a text editor (VS Code, Notepad++, or even plain Notepad)
2. Find the line that says `window.FIREBASE_CONFIG = null;`
3. Replace the **entire file contents** with the following, using YOUR values from Step 3:

```js
window.FIREBASE_CONFIG = {
  apiKey: "AIzaSyB...",
  authDomain: "shoppinglist-a1b2c.firebaseapp.com",
  projectId: "shoppinglist-a1b2c",
  storageBucket: "shoppinglist-a1b2c.firebasestorage.app",
  messagingSenderId: "123456789012",
  appId: "1:123456789012:web:abc123def456"
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

4. **Important:** Only replace the values inside the quotes (apiKey, authDomain, etc.) with YOUR values. Keep everything else exactly as shown above
5. Save the file

---

## Step 5: Deploy the App to the Internet (on your laptop)

This publishes your app so you can access it from your phones.

1. Open a **terminal** (on Windows: search for "Command Prompt" or "PowerShell" in the Start menu)
2. Run these commands one at a time:

```bash
npm install -g firebase-tools
```
This installs the Firebase command-line tool. You only need to do this once.

```bash
firebase login
```
A browser window will open asking you to sign in with Google. Sign in with the same Google account you used for Firebase, then come back to the terminal.

3. Now navigate to the app folder. If the repo is on your desktop:

```bash
cd Desktop/new_damn_repo/shopping-list-app
```

(Adjust the path to wherever the `shopping-list-app` folder actually is on your laptop)

4. Set up hosting:

```bash
firebase init hosting
```

It will ask you a series of questions. Answer them exactly like this:

| Question | Your answer |
|----------|-------------|
| **Please select an option:** | `Use an existing project` (use arrow keys, press Enter) |
| **Select a default Firebase project:** | Pick the project you created in Step 1 |
| **What do you want to use as your public directory?** | Type `.` (just a single dot) and press Enter |
| **Configure as a single-page app?** | Type `y` and press Enter |
| **Set up automatic builds and deploys with GitHub?** | Type `N` and press Enter |
| **File ./index.html already exists. Overwrite?** | Type `N` and press Enter |

5. Now deploy:

```bash
firebase deploy
```

6. When it finishes, it will show you a **Hosting URL** like:
```
https://shoppinglist-a1b2c.web.app
```

**This is your app's URL! Write it down or copy it.** This is what you'll open on your phones.

---

## Step 6: Add the App to Your Phone (on your Android phone)

1. Open **Chrome** on your phone (it must be Chrome, not Samsung Internet or another browser)
2. Type in the URL from Step 5 (e.g. `https://shoppinglist-a1b2c.web.app`)
3. The app should load and you'll see "The Shopping List" with a dark background
4. To add it to your home screen so it behaves like a real app:
   - Tap the **three-dot menu** (top-right corner of Chrome)
   - Tap **"Add to Home screen"** (or "Install app" if it shows that)
   - Tap **Add**
5. You'll now have an app icon on your home screen — tap it to open. It runs fullscreen like a proper app

---

## Step 7: Link Your Phones Together (on your phone, then WhatsApp)

This is what makes sure you and your wife both see the same list.

1. Open the app on your phone
2. Tap the **settings icon** in the top-right corner (it looks like a small sun/gear)
3. Scroll down to the **"Sync Status"** section
4. Tap the **"New"** button — this generates a random list ID (something like `k7x2m9p4`)
5. Tap **"Apply"** — the status should change to **"Connected & syncing"** in green
6. Tap **"Share via WhatsApp"** — this opens WhatsApp with a pre-written message containing:
   - The link to the app
   - The list ID
   - Instructions for your wife

Send it to your wife.

---

## Step 8: Your Wife Sets Up (on her Android phone)

Your wife needs to:

1. Open the WhatsApp message you sent
2. Tap the **link** in the message to open the app in Chrome
3. Add it to her home screen (same as Step 6 — three-dot menu → "Add to Home screen")
4. Open the app from her home screen
5. Tap the **settings icon** (top-right)
6. In the **List ID** field, paste the ID from the WhatsApp message
7. Tap **"Apply"**
8. It should say **"Connected & syncing"** in green

**That's it — you're both synced!** Anything either of you adds will appear on both phones instantly.

---

## How to Use the App

### Adding Items
- **By voice:** Tap the blue **microphone button** and say something like:
  - "tinned plum tomatoes Aldi"
  - "GF bread Sainsbury's"
  - "salted peanuts" (adds to whichever store tab you're currently on)
- **By typing:** Type in the text box at the bottom (e.g. `milk - Aldi`) and tap the **+** button

The app recognises store names even with variations — "Sainsbury's", "Sainsburys", "sainos" all work.

### Store Tabs
- Items are grouped by store — tap a tab to see just that store's items
- The **"All"** tab shows everything grouped by store
- To add a new store, either say its name when adding an item (e.g. "eggs Lidl" creates a Lidl tab), or tap the **+** button next to the tabs

### Ticking Items Off
- Tap the **circle** next to an item to tick it as bought
- A popup asks **"Where did you find this?"** — choose the aisle, which side, and how far along
- Tap **Skip** if you can't be bothered right now
- **The app remembers!** Next time you add that item to that store, it automatically shows where to find it

### The Aisle System
- **Aisle number:** The aisle number in the store (use the +/- buttons)
- **Side:** Left or Right as you walk into the aisle
- **Position:** Start (near where you entered the aisle), Middle, or End (far end)
- **For large stores like Sainsbury's** where aisles are split by a walkway down the middle: Go to Settings and toggle **"Split aisles"** on for that store. This adds a **Front half (A)** / **Back half (B)** choice — A is the half nearest the entrance, B is the far half

### After Shopping
Go to Settings → tap **"Remove all ticked items"** to clear everything you've bought, leaving unticked items for next time.
