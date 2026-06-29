# 🌙 Toddler Sleep Tracker

A simple, installable mobile web app (PWA) for tracking your toddler's daily sleep.
No accounts, no servers — data is stored on your phone and works fully offline.

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

It also shows a scrollable history and rolling 7-day averages, and lets you
export to CSV or back up / restore the data as JSON.

> Night sleep needs the **previous day's** bedtime to be logged, since the
> night spans two calendar dates. Keep logging each day and it fills in.

## How to install it on your phone

You need to open `index.html` over **https** (or `localhost`) for the
"add to home screen" / offline features to work.

**Quickest option — GitHub Pages**
1. Push this folder to GitHub.
2. Repo → Settings → Pages → deploy from your branch, `/toddler_sleep_tracker` folder.
3. Open the published URL on your phone:
   - **iPhone (Safari):** Share → *Add to Home Screen*.
   - **Android (Chrome):** menu ⋮ → *Install app* / *Add to Home Screen*.

**Try it locally on a computer**
```bash
cd toddler_sleep_tracker
python3 -m http.server 8000
# open http://localhost:8000
```

## Files
- `index.html` — the whole app (UI + logic, no build step)
- `manifest.webmanifest` — makes it installable
- `sw.js` — service worker for offline use
- `icon.svg` — app icon
