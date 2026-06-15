# Running OncAI Review on a Mac

This app isn't from the Mac App Store and isn't signed by Apple, so macOS asks
you to approve it **once**. After that it opens normally with a double-click.

> **Which file?** Use `oncai-review-<version>-macos-arm64.zip`. This is for
> **Apple Silicon** Macs (M1/M2/M3/M4 — any Mac from 2020 onward). To check:
> → menu → **About This Mac**; under **Chip** it should say "Apple ...".
> (If it says "Intel", let the sender know — you need a different build.)

## First time (one-time approval)

1. **Download** `oncai-review-<version>-macos-arm64.zip`.
2. **Double-click the `.zip`** in your Downloads to unzip it. You'll get
   **`oncai-review`** (an app with a blue clipboard-and-checkmark icon).
3. **Double-click `oncai-review`.** macOS will pop up a message like
   _"Apple could not verify 'oncai-review' is free of malware."_ Click
   **Done** (do **not** click "Move to Trash").
4. Open **System Settings** ( menu → System Settings).
5. Go to **Privacy & Security**, then scroll down to the **Security** section.
   You'll see a line: _"oncai-review" was blocked to protect your Mac._
   Click **Open Anyway**.
6. Confirm with **Touch ID** or your Mac password, then click **Open** in the
   final dialog.
7. **Two things open:** a small **Terminal** window (this is the app's engine —
   just leave it open) and your **web browser** with the review app. If macOS
   asks whether the app can access your **Documents** folder, click **Allow** —
   that's where your reviews are saved (`Documents/oncai_reviews/`).

That's it. **You only do steps 3–6 once.**

> **About the Terminal window:** it's normal. It shows the app's address
> (`http://localhost:...`) and stays open while the app runs. You don't need to
> type anything in it.

## Every time after that

- Double-click **`oncai-review`** — a Terminal window and your browser open with
  the app.
- In the app, click **Open a review package** and choose the
  `.review_pkg.json` file you were sent.
- **To quit, any of these work:**
  - Click the **Quit** button at the top-right of the app, **or**
  - Press **Ctrl-C** in the Terminal window, **or**
  - Close the Terminal window (confirm **Terminate**).
- **Closed the browser tab by accident?** Just **double-click `oncai-review`
  again** — it reconnects to the app that's still running and reopens your tab,
  right where you left off (your package stays loaded).

## Troubleshooting

- **"Open Anyway" isn't showing** in Privacy & Security — try double-clicking the
  app once first; the button only appears right after macOS blocks it.
- **Browser didn't open** — the Terminal window shows a `http://localhost:...`
  address; paste that into Safari/Chrome manually.
- **Still stuck?** Send a screenshot of what you see to the person who shared the
  app with you.
