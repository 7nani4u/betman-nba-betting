# 🏀 스포츠 베팅 엣지 파인더 (Sports Betting Edge Finder)

이 프로젝트는 머신러닝(ML) 모델을 기반으로 주요 스포츠 경기의 저평가된 베팅 라인을 식별하여 사용자에게 통계적 우위(엣지)를 제공하는 웹 애플리케이션입니다. Vercel 서버리스 환경에 최적화된 Flask 기반의 단일 페이지 애플리케이션(SPA)으로 구축되었습니다.

**🔗 라이브 데모:** [https://betman-betting.vercel.app](https://betman-betting.vercel.app)

---

## 🌟 주요 기능

*   **실시간 베팅 기회**: 농구, 축구, 야구 등 주요 스포츠 종목에 대한 베팅 기회를 실시간으로 제공합니다.
*   **상세 분석 데이터**: 각 베팅 기회에 대해 자체 모델 확률, 시장 확률, 엣지(Edge), 기대값(EV), 적정 베팅 금액(켈리 기준) 등 상세 분석 정보를 제공합니다.
*   **성과 대시보드**: 누적 수익, 승률, ROI 등 성과를 시각적으로 추적하는 분석 차트를 제공합니다.
*   **모델 정확도 비교**: 자체 모델의 예측 정확도를 실제 라스베가스 라인과 비교하여 모델의 우수성을 증명합니다.
*   **빠른 반응형 UI**: 페이지 새로고침 없이 종목을 선택하고 데이터를 동적으로 불러오는 SPA 아키텍처를 채택하여 사용자 경험을 극대화했습니다.

---

## 🏗️ 아키텍처 및 Vercel 최적화

이 애플리케이션은 Vercel의 서버리스 환경에서 안정적이고 빠르게 작동하도록 특별히 설계되었습니다. 초기 버전의 배포 실패 경험을 바탕으로 다음과 같은 아키텍처로 개선되었습니다.

| 항목 | 기술 스택 및 설계 | Vercel 최적화 이유 |
|---|---|---|
| **백엔드** | `Python`, `Flask` | Vercel의 Python 런타임과 완벽하게 호환되며, WSGI 표준을 준수하여 서버리스 함수로 쉽게 변환됩니다. |
| **프론트엔드** | `HTML`, `TailwindCSS`, `JavaScript (Fetch API)` | 정적 파일은 Vercel의 글로벌 CDN을 통해 매우 빠르게 제공됩니다. |
| **라우팅** | 단일 Flask 라우트 (`/`) + **쿼리 파라미터** (`?sport=농구`) | Vercel 서버리스 환경에서 한글 등 Non-ASCII 문자가 포함된 **URL 경로 파라미터** (`/농구`)가 손실되는 문제를 **쿼리 파라미터** 방식으로 변경하여 완벽하게 해결했습니다. |
| **데이터 로딩** | **AJAX (Asynchronous JavaScript and XML)** | 종목 선택 시 페이지 전체를 새로고침하는 대신, JavaScript의 `fetch` API를 사용해 `/api/sport` 엔드포인트에서 JSON 데이터만 비동기적으로 받아와 화면을 동적으로 업데이트합니다. 이를 통해 페이지 깜빡임이 없고 사용자 경험이 매우 빠릅니다. |
| **파일 구조** | `api/templates/` | Flask의 `render_template` 함수가 템플릿을 찾을 수 있도록, Vercel의 Python 런타임 표준에 맞춰 `templates` 폴더를 `api` 폴더 내부에 배치했습니다. |

---

## 📁 파일 구조

프로젝트는 Vercel 배포에 최적화된 단순하고 명확한 구조를 가집니다.

```
/
├── api/                  # Vercel 서버리스 함수 디렉터리
│   ├── index.py          # 메인 Flask 애플리케이션 (모든 로직 포함)
│   └── templates/
│       └── index.html    # 기본 HTML 템플릿
├── vercel.json           # Vercel 배포 및 라우팅 설정
└── requirements.txt      # Python 패키지 의존성 목록
```

---

## 🛠️ 로컬에서 실행하기

1.  **저장소 복제**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

3.  **의존성 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Flask 앱 실행**
    ```bash
    flask --app api/index run
    ```

5.  브라우저에서 [http://127.0.0.1:5000](http://127.0.0.1:5000) 주소로 접속합니다.

---

## 🚀 Vercel에 배포하기

1.  **GitHub 저장소에 코드 푸시**: 이 프로젝트의 모든 파일 (`api/`, `vercel.json`, `requirements.txt`)을 GitHub 저장소 루트에 푸시합니다.

2.  **Vercel 프로젝트 생성**:
    *   Vercel 대시보드에서 "Add New..." -> "Project"를 선택합니다.
    *   "Import Git Repository"에서 해당 GitHub 저장소를 선택하고 "Import" 버튼을 누릅니다.

3.  **배포 설정**: Vercel이 자동으로 `vercel.json` 파일을 인식하여 모든 설정을 구성합니다. 별도의 설정 변경 없이 "Deploy" 버튼을 누르면 배포가 시작됩니다.

4.  **완료**: 몇 분 안에 빌드 및 배포가 완료되고 고유한 URL이 생성됩니다.
