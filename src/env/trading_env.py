import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class BTCTradingEnv(gym.Env):
    """
    비트코인 트레이딩 Gymnasium 환경 (v3)

    v2 대비 개선사항:
    - [핵심] 액션 → 포지션 변화량(delta) 방식으로 재설계
      기존: action → 목표 포지션 절대값 (매도 경험 부족)
      변경: action → 현재 포지션에서의 변화량 (매수/매도 균등 학습)
    - 매도 시 실현 수익 보너스 추가
    - 에피소드 시작 시 초기 포지션 랜덤화 (다양한 상황 학습)

    Parameters
    ----------
    df               : 피처가 포함된 DataFrame (close 컬럼 필수)
    initial_balance  : 초기 자본 (USDT)
    fee_rate         : 거래 수수료율 (기본 0.05%)
    max_position     : 최대 포지션 비율 (기본 0.8)
    trade_threshold  : 거래 발동 최소 포지션 변화율 (기본 5%)
    position_penalty : 포지션 변화 페널티 강도 (기본 0.001)
    random_start_pos : 에피소드 시작 시 랜덤 포지션 여부 (기본 True)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df,
        initial_balance=10_000.0,
        fee_rate=0.0005,
        max_position=0.8,
        trade_threshold=0.05,
        position_penalty=0.001,
        random_start_pos=True,
    ):
        super().__init__()
        self.df               = df.reset_index(drop=True)
        self.initial_balance  = initial_balance
        self.fee_rate         = fee_rate
        self.max_position     = max_position
        self.trade_threshold  = trade_threshold
        self.position_penalty = position_penalty
        self.random_start_pos = random_start_pos

        assert 'close' in df.columns, "'close' 컬럼이 필요합니다."
        self.feature_cols  = [c for c in df.columns if c != 'close']
        self.n_features    = len(self.feature_cols)

        self.action_space  = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._reset_state()

    # ─────────────────────────────────────────────
    #  내부 상태 초기화
    # ─────────────────────────────────────────────
    def _reset_state(self):
        self.current_step     = 0
        self.balance          = self.initial_balance
        self.btc_held         = 0.0
        self.position_ratio   = 0.0
        self.avg_buy_price    = 0.0
        self.hold_steps       = 0
        self.total_fees       = 0.0
        self.trade_count      = 0
        self.portfolio_values = [self.initial_balance]
        self.peak_value       = self.initial_balance
        self.max_drawdown     = 0.0
        self.returns_history  = []

    # ─────────────────────────────────────────────
    #  포트폴리오 총 가치
    # ─────────────────────────────────────────────
    def _get_portfolio_value(self, price):
        return self.balance + self.btc_held * price

    # ─────────────────────────────────────────────
    #  Observation 구성
    # ─────────────────────────────────────────────
    def _get_observation(self):
        features      = self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        current_price = self.df.loc[self.current_step, 'close']
        unrealized_pnl = (
            (current_price - self.avg_buy_price) / self.avg_buy_price
            if self.avg_buy_price > 0 and self.btc_held > 0 else 0.0
        )
        position_info = np.array([
            self.position_ratio,
            np.clip(unrealized_pnl, -1.0, 1.0),
            np.clip(self.hold_steps / 100.0, 0, 1),
        ], dtype=np.float32)
        obs = np.concatenate([features, position_info])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ─────────────────────────────────────────────
    #  거래 실행 (★ 핵심 변경: delta 방식)
    # ─────────────────────────────────────────────
    def _execute_trade(self, action, current_price):
        """
        action [-1, 1] → 포지션 변화량(delta)으로 해석
          action = +1.0 → 최대한 매수 (현재 포지션 + max_position)
          action = -1.0 → 최대한 매도 (현재 포지션 - max_position)
          action =  0.0 → 홀드

        이 방식의 장점:
          포지션이 이미 있어도 매도 액션이 자연스럽게 작동
          매수/매도 경험을 균등하게 학습 가능
        """
        portfolio_val  = self._get_portfolio_value(current_price)
        current_ratio  = (self.btc_held * current_price) / (portfolio_val + 1e-9)

        # ★ action을 절대 목표가 아닌 변화량으로 해석
        delta          = action * self.max_position        # [-0.8, +0.8]
        target_ratio   = np.clip(current_ratio + delta, 0.0, self.max_position)
        delta_ratio    = target_ratio - current_ratio

        fee            = 0.0
        actual_delta   = 0.0
        realized_pnl   = 0.0

        if delta_ratio > self.trade_threshold:             # 매수
            buy_amount = min(portfolio_val * delta_ratio, self.balance)
            if buy_amount > 1.0:
                fee           = buy_amount * self.fee_rate
                btc_bought    = (buy_amount - fee) / current_price
                total_btc     = self.btc_held + btc_bought
                if total_btc > 0:
                    self.avg_buy_price = (
                        self.avg_buy_price * self.btc_held +
                        current_price * btc_bought
                    ) / total_btc
                self.btc_held    += btc_bought
                self.balance     -= buy_amount
                self.trade_count += 1
                self.hold_steps   = 0
                actual_delta      = delta_ratio

        elif delta_ratio < -self.trade_threshold:          # 매도
            sell_ratio  = min(abs(delta_ratio) / (current_ratio + 1e-9), 1.0)
            btc_to_sell = self.btc_held * sell_ratio
            if btc_to_sell * current_price > 1.0:
                sell_amount  = btc_to_sell * current_price
                fee          = sell_amount * self.fee_rate
                self.balance += sell_amount - fee
                self.btc_held -= btc_to_sell

                # ★ 실현 손익 계산 (매도 보너스용)
                if self.avg_buy_price > 0:
                    realized_pnl = (current_price - self.avg_buy_price) / self.avg_buy_price

                if self.btc_held < 1e-8:
                    self.btc_held      = 0.0
                    self.avg_buy_price = 0.0
                    self.hold_steps    = 0

                self.trade_count += 1
                actual_delta      = delta_ratio

        else:                                              # 홀드
            self.hold_steps += 1
            actual_delta     = 0.0

        self.total_fees    += fee
        new_val             = self._get_portfolio_value(current_price)
        self.position_ratio = (self.btc_held * current_price) / (new_val + 1e-9)

        return actual_delta, realized_pnl

    # ─────────────────────────────────────────────
    #  보상 함수 (★ 매도 수익 보너스 추가)
    # ─────────────────────────────────────────────
    def _compute_reward(self, prev_value, curr_value, actual_delta, fee_paid, realized_pnl):
        # 1) 로그 수익률
        log_return = np.log(curr_value / (prev_value + 1e-9))

        # 2) MDD 페널티 (20% 초과 시)
        if curr_value > self.peak_value:
            self.peak_value = curr_value
        drawdown          = (self.peak_value - curr_value) / (self.peak_value + 1e-9)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        drawdown_penalty  = max(0.0, (drawdown - 0.20)) * 2.0

        # 3) 포지션 변화 페널티 (잦은 거래 억제)
        position_pen = abs(actual_delta) * self.position_penalty

        # 4) 수수료 페널티
        fee_penalty  = (fee_paid / (curr_value + 1e-9)) * 10.0

        # 5) ★ 수익 실현 보너스 (이익 매도 장려)
        profit_bonus = max(0.0, realized_pnl) * 0.1

        self.returns_history.append(log_return)
        if len(self.returns_history) > 24:
            self.returns_history.pop(0)

        return float(log_return - drawdown_penalty - position_pen - fee_penalty + profit_bonus)

    # ─────────────────────────────────────────────
    #  Reset (★ 랜덤 초기 포지션 추가)
    # ─────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # ★ 랜덤 초기 포지션 (학습 환경에서만 True로 설정)
        if self.random_start_pos:
            init_price     = self.df.loc[0, 'close']
            init_position  = float(self.np_random.uniform(0.0, 0.5))
            buy_amount     = self.initial_balance * init_position
            self.btc_held       = buy_amount / init_price
            self.balance        = self.initial_balance - buy_amount
            self.avg_buy_price  = init_price
            self.position_ratio = init_position

        return self._get_observation(), {}

    # ─────────────────────────────────────────────
    #  Step
    # ─────────────────────────────────────────────
    def step(self, action):
        action        = float(np.clip(action, -1.0, 1.0))
        current_price = self.df.loc[self.current_step, 'close']
        prev_value    = self._get_portfolio_value(current_price)
        prev_fees     = self.total_fees

        # ★ 튜플 반환 (actual_delta, realized_pnl)
        actual_delta, realized_pnl = self._execute_trade(action, current_price)

        self.current_step += 1
        terminated    = self.current_step >= len(self.df) - 1
        next_price    = self.df.loc[self.current_step, 'close']
        curr_value    = self._get_portfolio_value(next_price)
        fee_paid      = self.total_fees - prev_fees

        self.portfolio_values.append(curr_value)

        if curr_value < self.initial_balance * 0.05:
            terminated = True

        reward = self._compute_reward(
            prev_value, curr_value, actual_delta, fee_paid, realized_pnl
        )

        info = {
            'step'           : self.current_step,
            'portfolio_value': curr_value,
            'balance'        : self.balance,
            'btc_held'       : self.btc_held,
            'position_ratio' : self.position_ratio,
            'current_price'  : next_price,
            'total_return'   : (curr_value - self.initial_balance) / self.initial_balance,
            'max_drawdown'   : self.max_drawdown,
            'trade_count'    : self.trade_count,
            'total_fees'     : self.total_fees,
        }
        return self._get_observation(), reward, terminated, False, info

    # ─────────────────────────────────────────────
    #  성과 시각화
    # ─────────────────────────────────────────────
    def render_performance(self, title='Trading Performance'):
        import platform
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        pv         = np.array(self.portfolio_values)
        prices     = self.df['close'].values[:len(pv)]
        bnh_values = self.initial_balance * (prices / prices[0])

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        axes[0].plot(pv, color='#627EEA', linewidth=1.2, label='RL 에이전트')
        axes[0].plot(bnh_values, color='#F7931A', linewidth=1.0,
                     alpha=0.7, linestyle='--', label='Buy & Hold')
        axes[0].axhline(self.initial_balance, color='gray', linestyle=':', linewidth=0.8)
        axes[0].set_title(title, fontsize=13, fontweight='bold')
        axes[0].set_ylabel('자산 (USDT)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        rolling_max = np.maximum.accumulate(pv)
        drawdown    = (rolling_max - pv) / (rolling_max + 1e-9)
        axes[1].fill_between(range(len(drawdown)), 0, -drawdown * 100,
                             color='red', alpha=0.4)
        axes[1].axhline(-20, color='red', linestyle='--', linewidth=0.8, label='경고선 -20%')
        axes[1].set_ylabel('낙폭 (%)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(prices, color='#F7931A', linewidth=0.8)
        axes[2].set_ylabel('BTC 가격 (USDT)')
        axes[2].set_xlabel('스텝 (1시간봉)')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        self._print_metrics(pv, bnh_values)

    def _print_metrics(self, pv, bnh_values):
        final_val   = pv[-1]
        total_ret   = (final_val - self.initial_balance) / self.initial_balance * 100
        bnh_ret     = (bnh_values[-1] - self.initial_balance) / self.initial_balance * 100
        returns     = np.diff(pv) / (pv[:-1] + 1e-9)
        sharpe      = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(24 * 365)
        rolling_max = np.maximum.accumulate(pv)
        max_dd      = ((rolling_max - pv) / (rolling_max + 1e-9)).max() * 100
        win_rate    = (returns > 0).mean() * 100

        print("\n" + "=" * 45)
        print("  📊 성과 요약")
        print("=" * 45)
        print(f"  최종 자산        : ${final_val:>12,.2f}")
        print(f"  총 수익률        : {total_ret:>+10.2f}%")
        print(f"  Buy & Hold       : {bnh_ret:>+10.2f}%")
        print(f"  알파             : {total_ret - bnh_ret:>+10.2f}%p")
        print("-" * 45)
        print(f"  샤프 지수        : {sharpe:>10.3f}")
        print(f"  최대 낙폭 (MDD)  : {max_dd:>10.2f}%")
        print(f"  승률             : {win_rate:>10.2f}%")
        print("-" * 45)
        print(f"  총 거래 횟수     : {self.trade_count:>10,}회")
        print(f"  총 수수료        : ${self.total_fees:>11,.2f}")
        print("=" * 45)
