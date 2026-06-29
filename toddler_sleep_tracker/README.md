# 🌙 Toddler Sleep Tracker

An installable mobile web app (PWA) for tracking your toddler's daily sleep,
with **real-time sync** between two phones (you + your partner) behind a shared
family passcode. Data lives in a free cloud database (Firebase Firestore), so
both phones always see the same, up-to-date entries — no stale data, no
cache-clearing.

## What you enter each day
- **Morning wake-up**
- **Nap start** / **Nap end**
- **Bedtime start** / **Fell asleep (night)**

## What it calculates automatically
| Metric | How it's worked out |
| --- | --- |
| **Night sleep** | Last night's *fell asleep* → this morning's *wake-up* (crosses midnight) |
| **Nap length** | Nap start → nap end |
| **Total sleep this day** | Night sleep + nap |
| **Wake window 1** | Morning wake → nap start |
| **Wake window 2** | Nap end → fell asleep at night |
| **Bedtime settling** | Bedtime start → fell asleep |

Plus scrollable history, rolling 7-day averages, and CSV/JSON export.

---

## One-time setup (≈10 min, do this once)

### 1. Create a free Firebase project
1. Go to <https://console.firebase.google.com> and sign in with Google.
2. **Add project** → name it anything (e.g. `toddler-sleep`) → you can disable
   Google Analytics → **Create project**.

### 2. Add a Web app & copy the config
1. In the project, click the **`</>`** (Web) icon to "Add an app".
2. Give it a nickname → **Register app**.
3. It shows a `firebaseConfig = { ... }` block. Copy those values.
4. Paste them into **`config.js`** in this folder (replace the `PASTE_…`
   placeholders). These keys are **not secret** — safe to commit.

### 3. Turn on Anonymous sign-in
1. Left menu → **Build → Authentication → Get started**.
2. **Sign-in method** tab → enable **Anonymous** → Save.

### 4. Create the database & set rules
1. Left menu → **Build → Firestore Database → Create database** →
   start in **production mode** → pick a location → Enable.
2. Open the **Rules** tab, replace everything with the rules below, **Publish**:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /rooms/{room}/{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

These rules require a signed-in (anonymous) session, and each family's data
lives under a `room` whose ID is derived from your passcode — so only people
with the passcode reach your data.

### 5. Deploy & open
- Deploy via GitHub Pages (repo **Settings → Pages**, branch + `/ (root)`),
  then open `…/toddler_sleep_tracker/` on each phone.
- On first open, enter the **same family passcode** on both phones.
- **iPhone (Safari):** Share → *Add to Home Screen*.
  **Android (Chrome):** menu ⋮ → *Install app*.

> Pick a decent passcode (not "1234") — anyone who knows it can read the data.
> To change it later, use **Sign out / change passcode** in the app. Starting a
> brand-new passcode starts a fresh, empty dataset.

---

## How sync works (why it won't go stale)
- The app subscribes to live database updates — when one phone saves, the other
  updates within ~a second, automatically.
- Offline, your entries are queued and sync the moment you're back online; the
  badge in the header shows **Synced / Saving… / Offline**.
- The app shell is served **network-first**, so app updates aren't stale either.

## Files
- `index.html` — the whole app (UI + logic, no build step)
- `config.js` — your Firebase keys (you fill these in)
- `manifest.webmanifest` — makes it installable
- `sw.js` — service worker (offline + network-first)
- `icon.svg` — app icon

> Note: night sleep needs the **previous day** logged, since the night spans two
> dates. It fills in as you keep logging each day.
