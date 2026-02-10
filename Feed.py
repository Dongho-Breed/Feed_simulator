import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# 1. 초기 데이터 설정 (세션 상태에 저장)
# -----------------------------------------------------------------------------
if 'feeds' not in st.session_state:
    st.session_state.feeds = [
        {"name": "알팔파", "cat": "조사료", "price": 900, "tdn": 52.5, "cp": 19.8, "ndf": 49.9},
        {"name": "IRG 사일리지", "cat": "조사료", "price": 350, "tdn": 37.6, "cp": 6.4, "ndf": 33.8},
        {"name": "볏짚", "cat": "조사료", "price": 200, "tdn": 39.0, "cp": 4.5, "ndf": 70.0},
        {"name": "옥수수", "cat": "농후사료", "price": 550, "tdn": 76.7, "cp": 7.2, "ndf": 8.4},
        {"name": "배합사료", "cat": "농후사료", "price": 650, "tdn": 70.0, "cp": 17.0, "ndf": 27.0},
        {"name": "TMR", "cat": "TMR", "price": 600, "tdn": 68.0, "cp": 14.0, "ndf": 32.0}
    ]

# -----------------------------------------------------------------------------
# 2. UI 구성
# -----------------------------------------------------------------------------
st.set_page_config(page_title="한우 사료 배합비 최적화", layout="wide")

# 제목 (요청하신 대로 소 이모티콘만 유지)
st.title("🐂 한우 사료 배합비 최적화 & 비용 분석기")
st.markdown("---")

# --- 사이드바: 원료 설정 ---
with st.sidebar:
    st.header("원료 및 단가 설정")
    
    # 카테고리별 단가 수정
    categories = ["조사료", "농후사료", "TMR"]
    updated_feeds = st.session_state.feeds.copy()
    
    with st.expander("단가 수정하기", expanded=False):
        for cat in categories:
            st.caption(f"[{cat}]")
            for i, feed in enumerate(updated_feeds):
                if feed['cat'] == cat:
                    new_price = st.number_input(
                        f"{feed['name']} (원)", value=feed['price'], step=10, key=f"price_{i}"
                    )
                    updated_feeds[i]['price'] = new_price
    st.session_state.feeds = updated_feeds

    st.markdown("---")
    
    # 선호 사료 설정
    st.subheader("선호 사료 우선 사용")
    st.info("특정 사료를 의무적으로 배합에 포함시킵니다.")
    
    feed_names = [f['name'] for f in st.session_state.feeds]
    priority_feeds = st.multiselect("우선 사용할 원료 선택", feed_names)
    
    min_ratio = 0.0
    if priority_feeds:
        min_ratio = st.slider("선택한 원료 최소 사용 비율 (%)", 1.0, 50.0, 10.0, step=1.0)
        st.caption(f"선택된 원료는 각각 최소 {min_ratio}% 이상 포함됩니다.")

# --- 메인 화면: 입력 폼 ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 사양 조건 입력")
    
    with st.container(border=True):
        st.write("개체 정보")
        c1, c2 = st.columns(2)
        avg_weight = c1.number_input("평균 체중 (kg)", value=450.0, step=10.0)
        weight_ratio = c2.number_input("체중비 (DMI율)", value=0.0211, step=0.001, format="%.4f")
        dmi = avg_weight * weight_ratio
        st.info(f"일일 목표 섭취량 (DMI): {dmi:.2f} kg")

    with st.container(border=True):
        st.write("영양소 목표치")
        target_tdn = st.number_input("TDN (에너지) % 이상", value=62.0, step=0.5)
        target_cp = st.number_input("CP (단백질) % 이상", value=12.0, step=0.5)
        target_ndf = st.number_input("NDF (섬유소) % 이상", value=35.0, step=0.5)

    with st.container(border=True):
        st.write("시장 상황")
        price_hike = st.slider("사료값 인상 시뮬레이션 (%)", 0, 50, 10)

# --- 최적화 로직 ---
def optimize_feed(feeds, targets, priority_list, min_r):
    prices = np.array([f['price'] for f in feeds])
    tdn = np.array([f['tdn'] for f in feeds])
    cp = np.array([f['cp'] for f in feeds])
    ndf = np.array([f['ndf'] for f in feeds])
    names = [f['name'] for f in feeds]
    n_feeds = len(feeds)

    # 목적 함수: 비용 최소화
    def objective(x):
        return np.dot(x, prices)

    # 제약 조건
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 100},
        {'type': 'ineq', 'fun': lambda x: np.dot(x, tdn) - targets['tdn'] * 100},
        {'type': 'ineq', 'fun': lambda x: np.dot(x, cp) - targets['cp'] * 100},
        {'type': 'ineq', 'fun': lambda x: np.dot(x, ndf) - targets['ndf'] * 100}
    ]
    
    # Bounds 설정 (선호 사료 적용)
    bounds = []
    for name in names:
        if name in priority_list:
            bounds.append((min_r, 100))
        else:
            bounds.append((0, 100))
            
    # 1차 시도
    x0 = [100/n_feeds] * n_feeds
    res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')
    
    return res, bounds

# --- 결과 출력 ---
with col2:
    st.subheader("2. 최적 배합 결과")
    
    if st.button("계산 실행 (Run)", type="primary", use_container_width=True):
        targets = {'tdn': target_tdn, 'cp': target_cp, 'ndf': target_ndf}
        
        # 1. 최적화 실행
        result, used_bounds = optimize_feed(st.session_state.feeds, targets, priority_feeds, min_ratio)
        
        feed_names = [f['name'] for f in st.session_state.feeds]
        prices = np.array([f['price'] for f in st.session_state.feeds])
        tdn_vals = np.array([f['tdn'] for f in st.session_state.feeds])
        cp_vals = np.array([f['cp'] for f in st.session_state.feeds])
        ndf_vals = np.array([f['ndf'] for f in st.session_state.feeds])

        is_priority_ignored = False
        
        # 2. 실패 시 로직
        if not result.success:
            if priority_feeds:
                retry_res, _ = optimize_feed(st.session_state.feeds, targets, [], 0)
                if retry_res.success:
                    result = retry_res
                    is_priority_ignored = True
                else:
                    result = retry_res 
            
        if not result.success:
            def error_objective(x):
                c_tdn = np.dot(x, tdn_vals) / 100
                c_cp = np.dot(x, cp_vals) / 100
                c_ndf = np.dot(x, ndf_vals) / 100
                loss = 0
                if c_tdn < targets['tdn']: loss += (targets['tdn'] - c_tdn)**2 * 100
                if c_cp < targets['cp']: loss += (targets['cp'] - c_cp)**2 * 100
                if c_ndf < targets['ndf']: loss += (targets['ndf'] - c_ndf)**2 * 100
                return loss

            cons_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 100}
            res_final = minimize(error_objective, [100/len(feed_names)]*len(feed_names), 
                                 bounds=[(0, 100)]*len(feed_names), constraints=cons_sum)
            ratios = res_final.x
            status_type = "FAIL"
        else:
            ratios = result.x
            status_type = "SUCCESS"

        # --- 메시지 출력 ---
        if status_type == "SUCCESS":
            if is_priority_ignored:
                st.warning(f"선호하신 원료({', '.join(priority_feeds)})를 {min_ratio}% 이상 쓰면 영양소 기준을 맞출 수 없어, 선호 조건을 제외하고 최적화했습니다.")
            else:
                st.success("모든 조건(영양소 + 선호 원료)을 만족하는 최적 배합비입니다.")
        else:
            st.error("현재 원료로는 영양소 기준을 달성할 수 없습니다.")
            st.caption("아래 결과는 목표치에 가장 근접한 수치입니다.")

        # --- 수치 계산 ---
        final_ratios = np.round(ratios, 2)
        final_tdn = np.dot(final_ratios, tdn_vals) / 100
        final_cp = np.dot(final_ratios, cp_vals) / 100
        final_ndf = np.dot(final_ratios, ndf_vals) / 100
        
        feed_amounts = dmi * (final_ratios / 100)
        daily_cost = np.dot(feed_amounts, prices)
        increased_cost = daily_cost * (1 + price_hike/100)

        # 1) 영양소
        st.markdown("#### 영양소 충족률")
        c1, c2, c3 = st.columns(3)
        def show_metric(col, label, val, target):
            diff = val - target
            col.metric(label, f"{val:.1f}%", f"{diff:.1f}%", delta_color="normal" if diff >= -0.05 else "inverse")
        
        show_metric(c1, "TDN", final_tdn, target_tdn)
        show_metric(c2, "CP", final_cp, target_cp)
        show_metric(c3, "NDF", final_ndf, target_ndf)

        # 2) 배합비 표
        st.markdown("#### 추천 배합 설계")
        df_res = pd.DataFrame({
            "원료명": feed_names,
            "비율(%)": final_ratios,
            "급여량(kg)": feed_amounts,
            "단가": prices,
            "비용(원)": feed_amounts * prices
        })
        df_res = df_res[df_res["비율(%)"] > 0.01].sort_values("비율(%)", ascending=False)
        
        # 선호 사료 하이라이트 (글자색 검정 적용됨)
        def highlight_priority(row):
            if row['원료명'] in priority_feeds and not is_priority_ignored and status_type == "SUCCESS":
                return ['background-color: #e6f3ff; color: #000000'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_res.style.apply(highlight_priority, axis=1).format({
                "비율(%)": "{:.1f}", "급여량(kg)": "{:.2f}", "비용(원)": "{:,.0f}"
            }), 
            use_container_width=True, 
            hide_index=True
        )

        # 3) 비용
        st.markdown("#### 경제성 분석 (1일/두)")
        ec1, ec2 = st.columns(2)
        ec1.metric("현재 비용", f"{int(daily_cost):,}원")
        ec2.metric(f"단가 {price_hike}% 상승 시", f"{int(increased_cost):,}원", f"+{int(increased_cost-daily_cost):,}원", delta_color="inverse")

# -----------------------------------------------------------------------------
# 3. 하단 정보 섹션 (계산식 및 원료 정보)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("참고: 원료 성분 및 계산 산식")

info_col1, info_col2 = st.columns([1, 1])

with info_col1:
    st.subheader("1. 원료별 영양소 기준 (국립축산과학원)")
    df_info = pd.DataFrame(st.session_state.feeds)
    st.dataframe(
        df_info[['name', 'tdn', 'cp', 'ndf']].rename(
            columns={'name': '원료명', 'tdn': 'TDN(%)', 'cp': 'CP(%)', 'ndf': 'NDF(%)'}
        ),
        hide_index=True,
        use_container_width=True
    )

with info_col2:
    st.subheader("2. 계산 산식 (Formula)")
    st.markdown("""
    **① 일일 섭취량 (DMI)**
    $$ DMI(kg) = 체중(kg) \\times 체중비 $$
    
    **② 혼합 영양소 함량 (%)**
    $$ \\text{Nutrient}(\\%) = \\sum \\left( \\text{각 원료 배합비율} \\times \\text{원료 성분함량} \\right) \\div 100 $$
    *(예: 혼합 TDN = 각 원료의 TDN 기여분의 합계)*
    
    **③ 일일 사료비 (원)**
    $$ \\text{Daily Cost} = \\sum \\left( DMI \\times \\frac{\\text{배합비율}}{100} \\times \\text{단가} \\right) $$
    """)
    