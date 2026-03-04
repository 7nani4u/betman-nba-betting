
# Vercel 배포 오류 분석 및 해결 보고서

## 1. 문제 분석

제공해주신 `betman-betting.vercel.app` URL에서 발생하는 **404 NOT_FOUND** 오류는 Vercel 플랫폼이 애플리케이션을 정상적으로 실행하고 요청된 경로에 연결할 수 없음을 의미합니다. 스크린샷의 오류 메시지는 Vercel의 라우팅 시스템이 해당 요청을 처리할 수 있는 서버리스 기능(Serverless Function)을 찾지 못했음을 나타냅니다.

根本적인 원인은 기존 `dashboard.py` 코드가 **Streamlit** 프레임워크를 사용하여 작성되었기 때문입니다. Streamlit은 자체 웹 서버를 실행하여 상태를 유지하는 방식으로 동작하므로, 요청-응답 모델 기반의 Vercel 서버리스 환경과 직접적으로 호환되지 않습니다. Vercel은 Python 백엔드를 배포할 때, **WSGI(Web Server Gateway Interface)** 또는 **ASGI(Asynchronous Server Gateway Interface)** 표준을 따르는 웹 프레임워크(예: Flask, FastAPI)를 기대합니다 [1].

## 2. 해결 방안

이 문제를 해결하기 위해, 기존 Streamlit 애플리케이션의 로직과 UI를 유지하면서 Vercel의 서버리스 아키텍처와 호환되는 **Flask** 프레임워크로 코드를 재작성했습니다. Flask는 경량 WSGI 프레임워크로, Vercel에서 공식적으로 지원하며 서버리스 함수로 쉽게 변환할 수 있습니다 [2].

수정된 애플리케이션은 다음과 같은 구조를 가집니다.

```
/betman-app
├── api/
│   └── index.py         # Flask 애플리케이션 로직
├── templates/
│   └── index.html       # 프론트엔드 HTML 템플릿
├── vercel.json          # Vercel 배포 설정 파일
└── requirements.txt     # Python 의존성 목록
```

### 3. 주요 변경 사항

#### 3.1. `api/index.py` (Flask 애플리케이션)

- 기존 `dashboard.py`의 데이터 처리 및 Plotly 차트 생성 함수들은 대부분 그대로 유지했습니다.
- Streamlit의 UI 컴포넌트(`st.title`, `st.sidebar`, `st.metric` 등)를 Flask의 라우트(`@app.route("/")`)와 HTML 템플릿 렌더링(`render_template`) 방식으로 대체했습니다.
- Plotly 차트를 JSON 형식으로 변환하여 HTML 템플릿으로 전달하고, 프론트엔드에서 JavaScript(Plotly.js)를 사용해 렌더링하도록 수정했습니다. 이는 서버와 클라이언트 간의 명확한 역할 분리를 가능하게 합니다.

#### 3.2. `templates/index.html` (프론트엔드)

- Jinja2 템플릿 언어를 사용하여 동적으로 데이터를 표시합니다.
- **TailwindCSS**를 사용하여 기존 Streamlit 앱과 유사한 깔끔하고 반응형인 UI를 신속하게 구현했습니다.
- Plotly.js 라이브러리를 포함하여 백엔드에서 전달된 차트 JSON 데이터를 시각화합니다.
- 탭 전환과 같은 동적인 UI 상호작용은 간단한 JavaScript로 처리됩니다.

#### 3.3. `vercel.json` (배포 설정)

이 파일은 Vercel에 프로젝트를 어떻게 빌드하고 요청을 라우팅할지 알려주는 핵심 설정 파일입니다.

```json
{
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

- **builds**: `api/index.py` 파일을 `@vercel/python` 런타임을 사용하여 빌드하도록 지정합니다. 이 과정에서 Vercel은 Flask 애플리케이션을 단일 서버리스 함수로 패키징합니다.
- **routes**: 모든 들어오는 요청(`/(.*)`)을 `api/index.py`에서 생성된 서버리스 함수로 전달하도록 설정합니다. 이로써 사용자가 웹사이트에 접속했을 때 Flask 앱이 응답하게 됩니다.

#### 3.4. `requirements.txt` (의존성)

- `streamlit`을 제거하고 `Flask`를 추가했습니다.
- 원본 파일에 포함되어 있었지만 실제 코드에서는 사용되지 않는 `xgboost`, `scikit-learn`, `joblib` 라이브러리는 주석 처리했습니다. 이는 서버리스 함수의 용량을 줄여 배포 속도를 높이고 비용을 절감하는 데 도움이 됩니다.

## 4. 배포 방법

1. 첨부된 `betman-app.zip` 파일의 압축을 풉니다.
2. 압축 해제된 `betman-app` 폴더를 GitHub 리포지토리의 루트에 업로드합니다.
3. Vercel 대시보드에서 해당 GitHub 리포지토리를 연결하여 프로젝트를 다시 배포합니다.

이제 Vercel은 `vercel.json` 설정에 따라 프로젝트를 올바르게 빌드하고 Flask 애플리케이션을 실행하여 웹사이트가 정상적으로 표시될 것입니다.

---

### 참고 자료
[1] Vercel. (2026, January 30). *Using the Python Runtime with Vercel Functions*. Vercel Docs. https://vercel.com/docs/functions/runtimes/python
[2] Vercel. (2025, November 13). *Flask on Vercel*. Vercel Docs. https://vercel.com/docs/frameworks/backend/flask
