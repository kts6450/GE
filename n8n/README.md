# n8n 설정 방법

## 포트(자동)

**`start-n8n.cmd` / `start-n8n.ps1`** 는 **듣는 포트가 비어 있을 때까지** `16888` → `5678` → 기타 백업 →(전부 잡이면) **빈 포트 자동** 순으로 `N8N_PORT` 를 정한 뒤, **`N8N_DISABLE_UI=false`** 로 `n8n start` 를 실행합니다. 콘솔에 **`http://127.0.0.1:포트/`** 를 크게 찍고, 직전에 쓴 포트는 `n8n/.n8n-ui-port` 에 한 줄로 남깁니다.

- **접속:** 콘솔에 나온 주소(대부분 `16888` 또는 5678이 비었을 때 `5678`)
- **TCube** 가 5678을 **http.sys** 로 쓰면 `Get-NetTCPConnection` 상으로는 5678에 리스너가 있어 **다음 후보(16888 등)**로 넘어갑니다. `net stop` 이 오류 5로 막혀도 **스크립트만으로 n8n을 띄울 수 있게** 이렇게 잡아 둡니다.
- **수동:** 그냥 `n8n start` (공식 기본 5678) — 5678이 막힌 PC는 실패하므로 위 스크립트 사용

### TCube·5678을 정리하는 경우(가능한 PC만)

`netsh http show servicestate`에 **`HTTP://127.0.0.1:5678` … `/TCUBE/`** 가 보이면 5678은 TCube 쪽입니다. **관리자** 셸에서 `Stop-Service TCube` → 성공하면 일반 셸에서 `n8n start` (기본 5678). 실패(오류 5 등)는 **이 프로젝트의 16888 스크립트**로 진행하세요.

**참고:** WSL, Hyper-V NAT의 **excludedportrange** 는 `netsh int ipv4 show excludedportrange protocol=tcp` 로 별도입니다.

### 브라우저에 `Cannot GET /` (빈 화면만)

n8n **2.x**에서는 **`N8N_DISABLE_UI=true`**이면 `index.html`이 안 만들어지고 **`GET /`가 404**로 떨어집니다(Express `Cannot GET /`). 전역/사용자 환경 변수에 켜 두었거나 `.env`에 있으면 그렇게 될 수 있어, **`start-n8n.ps1`은 `N8N_DISABLE_UI`를 지운 뒤 `0`으로 덮고**, `N8N_PATH`는 **`/`(루트)** 로 맞춥니다.

- **필수:** 기존 n8n·node가 그 포트를 쓰는 중이면 **프로세스를 끄고** 다시 `start-n8n` 실행(다른 앱이 16888만 응답하는 경우에도 이 메시지가 남).
- **`/signin`**: 루트가 비면 `http://127.0.0.1:포트/signin` 도 시도.
- `curl -I` 는 **HEAD** — 브라우저와 달리 404로 보일 수 있음. **GET**은 `curl.exe` 로 확인.

---

1. n8n을 위 방법으로 실행합니다.
2. `workflow_export.json`을 Import합니다.
3. `Run GE Prediction Script` 노드의 경로가 현재 프로젝트 경로와 맞는지 확인합니다.
4. Google Sheets 노드에서 Spreadsheet ID와 Credential을 입력합니다.
5. Telegram 노드에서 Bot Credential과 Chat ID를 입력합니다.
6. 비활성화된 Google Sheets, Telegram 노드를 활성화합니다.
7. 수동 실행으로 정상 동작을 확인한 뒤 워크플로우를 Active로 변경합니다.

Google Sheets에 권장하는 헤더:

```text
generated_at_utc,last_market_date,ticker,model,current_close,predicted_next_close,predicted_next_return_pct,rsi_14,volatility_20_pct
```

Telegram 대신 Email이나 Discord를 쓰고 싶다면 `Parse Prediction Result` 뒤에 해당 노드를 연결하고 `telegram_message` 값을 본문으로 사용하면 됩니다.
