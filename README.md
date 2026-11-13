# Indian Stock Intelligence

A small Streamlit dashboard for exploring NSE/BSE share prices, pullback context, selected fundamentals, and a historical machine-learning evaluation.

## Important limits

- Yahoo Finance data accessed through `yfinance` is not an official NSE feed and is not guaranteed to be live, complete, or available. It can be delayed, corrected, rate-limited, or changed without notice. Verify data with exchange and company sources.
- `yfinance` describes its project as an unaffiliated research/education tool and says Yahoo Finance data is intended for personal use. Review Yahoo's terms before using this app beyond personal research.
- A lower share price does not mean a company is undervalued. Fundamental fields may be stale, unavailable, or defined inconsistently. The checklist does not replace reading filings or comparing a company with its industry.
- The model's target is a demonstration: adjusted close more than 5% higher after 20 trading sessions. It uses four expanding, chronological walk-forward folds with a 20-session gap before each test fold. The final 20 dates with unknown outcomes are excluded. Results are historical, single-stock results, not a forecast or a recommendation. Transaction costs, taxes, liquidity, survivorship bias, and changing market conditions are not modeled.
- Nothing in the dashboard guarantees returns or is personalized buy/sell advice.

## Run it on your computer (Windows, beginner steps)

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/). During setup, tick **Add Python to PATH**.
2. Download this folder, or open it in File Explorer. Click the address bar, type `powershell`, and press Enter.
3. Create a private environment for the app by typing:

   ```powershell
   py -3.11 -m venv .venv
   ```

4. Turn that environment on:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If Windows blocks this command, open Command Prompt in this folder and use `.venv\Scripts\activate.bat` instead.

5. Install the app's ingredients:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   python -m streamlit run stock_dashboard.py
   ```

   A browser page should open. If it does not, copy the `http://localhost:8501` address shown in the terminal into your browser. Leave the terminal window open while using the dashboard.

7. In the left box, enter an NSE symbol such as `RELIANCE` or `RELIANCE.NS`. The app adds `.NS` automatically when no exchange suffix is given. Use `.BO` for BSE. Yahoo's symbol list can change, so confirm any symbol that fails.

## Put it on Streamlit Community Cloud

You will need a GitHub account. Streamlit Community Cloud deploys from a GitHub repository, and this folder already contains `requirements.txt`.

1. On [GitHub](https://github.com/), create a new repository named `indian-stock-intelligence`.
2. Upload the **contents of this folder** to the repository's top level. The top level should show `stock_dashboard.py`, `stock_tools.py`, and `requirements.txt`.
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in using GitHub.
4. Choose **Create app**, then select your repository, the `main` branch, and `stock_dashboard.py` as the app file.
5. Choose **Deploy**. The first start can take a few minutes while the libraries install. Streamlit displays your app's web address when it is ready.
6. To update the app later, upload or save changed files to GitHub. Streamlit will redeploy the app.

You do not need an API key for this sample app. Do not put passwords or private account details in the code or repository. Community Cloud uses the Python version it supports; Python 3.11 is the local starting point recommended here.

## Common fixes

- **No prices returned:** check ticker spelling and exchange suffix (`.NS` for NSE, `.BO` for BSE); try again later if Yahoo is unavailable.
- **Some fundamentals say Not provided:** the upstream source does not provide every field for every company. Do not treat missing data as a pass.
- **Model says not enough data:** use a longer price history, or accept that a historical evaluation is not available for that share.
- **App will not start after editing:** in the app folder, activate `.venv` and rerun `python -m pip install -r requirements.txt`.

