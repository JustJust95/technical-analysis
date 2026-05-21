import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pattern Recognition Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .pattern-card {
        background: #161b22;
        border-left: 4px solid;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .bullish { border-left-color: #28a745 !important; }
    .bearish { border-left-color: #dc3545 !important; }
    .neutral { border-left-color: #fd7e14 !important; }
    
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 4px;
    }
    
    .badge-green { background: #1a4731; color: #28a745; }
    .badge-red { background: #3d1a22; color: #dc3545; }
    .badge-orange { background: #3d2a1a; color: #fd7e14; }
    .badge-blue { background: #1a2a3d; color: #58a6ff; }
    
    h1, h2, h3 { color: #e6edf3 !important; }
    .stSelectbox label, .stTextInput label { color: #e6edf3 !important; }
    
    hr { border-color: #30363d; }
    
    .disclaimer {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        font-size: 12px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PatternSignal:
    name: str
    type: str
    status: str
    direction: str
    confidence: float
    quality_score: int
    entry_zone: Tuple[float, float]
    stop_loss: float
    targets: List[float]
    strategy: str
    timeframe: str
    description: str
    warnings: List[str]
    pattern_points: Optional[List[Tuple[int, float]]] = None
    trendlines: Optional[List[Tuple[int, float, int, float]]] = None
    neckline: Optional[float] = None
    apex_idx: Optional[int] = None
    pattern_indices: Optional[List[int]] = None


# ============================================================
# PATTERN RECOGNIZER CLASS
# ============================================================

class AdvancedPatternRecognizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.patterns_found = []
        self.market_context = self._analyze_market_context()

    def _analyze_market_context(self) -> dict:
        context = {}
        if len(self.df) < 50:
            return {'trend': 'insufficient_data', 'volatility': 'unknown', 'strength': 'unknown'}

        sma_20 = self.df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = self.df['Close'].rolling(50).mean().iloc[-1]
        current_price = self.df['Close'].iloc[-1]

        if current_price > sma_20 > sma_50:
            context['trend'] = 'bullish'
        elif current_price < sma_20 < sma_50:
            context['trend'] = 'bearish'
        else:
            context['trend'] = 'neutral'

        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = true_range.rolling(14).mean()

        atr_pct = self.df['ATR'].iloc[-1] / current_price * 100
        context['volatility'] = 'high' if atr_pct > 3 else 'normal' if atr_pct > 1.5 else 'low'

        self.df['Volume_SMA'] = self.df['Volume'].rolling(20).mean()
        vol_ratio = self.df['Volume'].iloc[-1] / self.df['Volume_SMA'].iloc[-1]
        context['volume'] = 'high' if vol_ratio > 1.5 else 'normal' if vol_ratio > 0.7 else 'low'
        context['current_price'] = current_price
        context['sma_20'] = sma_20
        context['sma_50'] = sma_50
        return context

    def _find_swing_points(self, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.df)
        window = min(window, n // 4)
        if window < 2:
            window = 2
        high_idx = argrelextrema(self.df['High'].values, np.greater, order=window)[0]
        low_idx = argrelextrema(self.df['Low'].values, np.less, order=window)[0]
        return high_idx, low_idx

    def detect_head_and_shoulders(self) -> List[PatternSignal]:
        patterns = []
        if len(self.df) < 60:
            return patterns
        window = min(5, len(self.df) // 12)
        if window < 2:
            return patterns
        high_idx, low_idx = self._find_swing_points(window)
        if len(high_idx) < 3 or len(low_idx) < 3:
            return patterns
        highs_values = self.df['High'].values[high_idx[-10:]]
        lows_values = self.df['Low'].values[low_idx[-10:]]

        if len(highs_values) >= 3 and len(lows_values) >= 2:
            for i in range(len(highs_values) - 2):
                left_shoulder = highs_values[i]
                head = highs_values[i + 1]
                right_shoulder = highs_values[i + 2]
                if head > left_shoulder and head > right_shoulder:
                    height_diff = abs(left_shoulder - right_shoulder) / head
                    if height_diff < 0.15:
                        neckline_lows = lows_values[i:i+3] if len(lows_values) > i+2 else lows_values[i:]
                        if len(neckline_lows) >= 2:
                            neckline = np.mean(neckline_lows)
                            current_price = self.df['Close'].iloc[-1]
                            left_idx = high_idx[-10:][i]
                            head_idx = high_idx[-10:][i+1]
                            right_idx = high_idx[-10:][i+2]
                            pattern_points = [
                                (left_idx, left_shoulder, 'LS', 'Left Shoulder'),
                                (head_idx, head, 'H', 'Head'),
                                (right_idx, right_shoulder, 'RS', 'Right Shoulder')
                            ]
                            status = 'completed' if current_price < neckline else 'forming'
                            confidence = 0.8 if status == 'completed' else 0.5
                            target = neckline - (head - neckline)
                            patterns.append(PatternSignal(
                                name="Head and Shoulders TOP",
                                type="reversal", status=status, direction="bearish",
                                confidence=confidence,
                                quality_score=self._calculate_quality_score(status, confidence, 'bearish'),
                                entry_zone=(neckline * 0.99, neckline * 1.01),
                                stop_loss=head * 1.02,
                                targets=[target, target * 0.95],
                                strategy=self._generate_hns_strategy(status, 'top'),
                                timeframe="Swing (2-8 settimane)",
                                description=f"Pattern ribassista. Testa ${head:.2f}, spalle ~${left_shoulder:.2f}",
                                warnings=self._generate_warnings(status, 'reversal'),
                                pattern_points=pattern_points,
                                neckline=neckline,
                                pattern_indices=[left_idx, head_idx, right_idx]
                            ))

        if len(lows_values) >= 3:
            for i in range(len(lows_values) - 2):
                left_shoulder = lows_values[i]
                head = lows_values[i + 1]
                right_shoulder = lows_values[i + 2]
                if head < left_shoulder and head < right_shoulder:
                    height_diff = abs(left_shoulder - right_shoulder) / abs(head)
                    if height_diff < 0.15:
                        neckline_highs = highs_values[i:i+3] if len(highs_values) > i+2 else highs_values[i:]
                        if len(neckline_highs) >= 2:
                            neckline = np.mean(neckline_highs)
                            current_price = self.df['Close'].iloc[-1]
                            left_idx = low_idx[-10:][i]
                            head_idx = low_idx[-10:][i+1]
                            right_idx = low_idx[-10:][i+2]
                            pattern_points = [
                                (left_idx, left_shoulder, 'LS', 'Left Shoulder'),
                                (head_idx, head, 'H', 'Head'),
                                (right_idx, right_shoulder, 'RS', 'Right Shoulder')
                            ]
                            status = 'completed' if current_price > neckline else 'forming'
                            confidence = 0.8 if status == 'completed' else 0.5
                            target = neckline + (neckline - head)
                            patterns.append(PatternSignal(
                                name="Inverse Head and Shoulders",
                                type="reversal", status=status, direction="bullish",
                                confidence=confidence,
                                quality_score=self._calculate_quality_score(status, confidence, 'bullish'),
                                entry_zone=(neckline * 0.99, neckline * 1.01),
                                stop_loss=head * 0.98,
                                targets=[target, target * 1.05],
                                strategy=self._generate_hns_strategy(status, 'bottom'),
                                timeframe="Swing (2-8 settimane)",
                                description=f"Pattern rialzista. Testa ${head:.2f}",
                                warnings=self._generate_warnings(status, 'reversal'),
                                pattern_points=pattern_points,
                                neckline=neckline,
                                pattern_indices=[left_idx, head_idx, right_idx]
                            ))
        return patterns

    def detect_double_top_bottom(self) -> List[PatternSignal]:
        patterns = []
        if len(self.df) < 30:
            return patterns
        window = min(3, len(self.df) // 10)
        if window < 2:
            return patterns
        high_idx, low_idx = self._find_swing_points(window)

        if len(high_idx) >= 2:
            highs = self.df['High'].values[high_idx[-5:]]
            for i in range(len(highs) - 1):
                if abs(highs[i] - highs[i+1]) / highs[i] < 0.03:
                    current_price = self.df['Close'].iloc[-1]
                    valley = self.df['Low'].values[low_idx[-10:]].min() if len(low_idx) > 0 else current_price * 0.95
                    idx1 = high_idx[-5:][i]
                    idx2 = high_idx[-5:][i+1]
                    pattern_points = [(idx1, highs[i], 'T1', 'First Top'), (idx2, highs[i+1], 'T2', 'Second Top')]
                    status = 'completed' if current_price < valley else 'forming'
                    confidence = 0.85 if status == 'completed' else 0.55
                    target = valley - (max(highs[i], highs[i+1]) - valley)
                    patterns.append(PatternSignal(
                        name="Double Top", type="reversal", status=status, direction="bearish",
                        confidence=confidence,
                        quality_score=self._calculate_quality_score(status, confidence, 'bearish'),
                        entry_zone=(valley * 0.99, valley * 1.01), stop_loss=max(highs[i], highs[i+1]) * 1.02,
                        targets=[target], strategy=self._generate_double_strategy(status, 'top'),
                        timeframe="Swing (1-4 settimane)",
                        description=f"Doppio massimo ~${highs[i]:.2f}. Supporto ${valley:.2f}",
                        warnings=self._generate_warnings(status, 'reversal'),
                        pattern_points=pattern_points, neckline=valley, pattern_indices=[idx1, idx2]
                    ))
                    break

        if len(low_idx) >= 2:
            lows = self.df['Low'].values[low_idx[-5:]]
            for i in range(len(lows) - 1):
                if abs(lows[i] - lows[i+1]) / abs(lows[i]) < 0.03:
                    current_price = self.df['Close'].iloc[-1]
                    peak = self.df['High'].values[high_idx[-10:]].max() if len(high_idx) > 0 else current_price * 1.05
                    idx1 = low_idx[-5:][i]
                    idx2 = low_idx[-5:][i+1]
                    pattern_points = [(idx1, lows[i], 'B1', 'First Bottom'), (idx2, lows[i+1], 'B2', 'Second Bottom')]
                    status = 'completed' if current_price > peak else 'forming'
                    confidence = 0.85 if status == 'completed' else 0.55
                    target = peak + (peak - min(lows[i], lows[i+1]))
                    patterns.append(PatternSignal(
                        name="Double Bottom", type="reversal", status=status, direction="bullish",
                        confidence=confidence,
                        quality_score=self._calculate_quality_score(status, confidence, 'bullish'),
                        entry_zone=(peak * 0.99, peak * 1.01), stop_loss=min(lows[i], lows[i+1]) * 0.98,
                        targets=[target], strategy=self._generate_double_strategy(status, 'bottom'),
                        timeframe="Swing (1-4 settimane)",
                        description=f"Doppio minimo ~${lows[i]:.2f}. Resistenza ${peak:.2f}",
                        warnings=self._generate_warnings(status, 'reversal'),
                        pattern_points=pattern_points, neckline=peak, pattern_indices=[idx1, idx2]
                    ))
                    break
        return patterns

    def detect_triangle_patterns(self) -> List[PatternSignal]:
        patterns = []
        if len(self.df) < 30:
            return patterns
        lookback = min(60, len(self.df))
        recent_df = self.df.iloc[-lookback:]
        highs = recent_df['High'].values
        lows = recent_df['Low'].values
        x = np.arange(len(highs))
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        high_slope = high_coef[0]
        low_slope = low_coef[0]
        current_price = self.df['Close'].iloc[-1]
        high_line = high_coef[1] + high_slope * x
        low_line = low_coef[1] + low_slope * x

        if high_slope < -0.001 and low_slope > 0.001:
            pattern_points = [
                (len(self.df) - lookback, high_line[0], 'RH', 'Resistance Start'),
                (len(self.df) - 1, high_line[-1], 'RE', 'Resistance End'),
                (len(self.df) - lookback, low_line[0], 'SH', 'Support Start'),
                (len(self.df) - 1, low_line[-1], 'SE', 'Support End')
            ]
            direction = 'bullish' if self.market_context['trend'] == 'bullish' else 'bearish'
            target = current_price + (max(highs) - min(lows)) if direction == 'bullish' else current_price - (max(highs) - min(lows))
            patterns.append(PatternSignal(
                name="Symmetrical Triangle", type="bilateral", status='forming', direction=direction,
                confidence=0.5, quality_score=self._calculate_quality_score('forming', 0.5, direction),
                entry_zone=(lows[-1] * 0.99, highs[-1] * 1.01),
                stop_loss=min(lows) * 0.98 if direction == 'bullish' else max(highs) * 1.02,
                targets=[target], strategy="Attendere breakout con volume. Entrare nella direzione del breakout",
                timeframe="Swing/Daytrading", description="Triangolo simmetrico in compressione",
                warnings=["Pattern bilaterale: attendere conferma breakout"],
                pattern_points=pattern_points
            ))

        elif abs(high_slope) < 0.002 and low_slope > 0.001:
            resistance = np.mean(highs[-10:])
            status = 'completed' if current_price > resistance else 'forming'
            confidence = 0.85 if status == 'completed' else 0.6
            target = resistance + (resistance - min(lows))
            patterns.append(PatternSignal(
                name="Ascending Triangle", type="continuation", status=status, direction="bullish",
                confidence=confidence, quality_score=self._calculate_quality_score(status, confidence, 'bullish'),
                entry_zone=(resistance * 0.99, resistance * 1.01), stop_loss=min(lows[-10:]) * 0.98,
                targets=[target], strategy="Breakout rialzista atteso sopra la resistenza",
                timeframe="Swing (settimane)", description=f"Triangolo ascendente. Resistenza ${resistance:.2f}",
                warnings=["Attendere conferma volume al breakout"],
                pattern_points=[(len(self.df) - lookback, resistance, 'R', 'Resistance'),
                                (len(self.df) - 1, low_line[-1], 'S', 'Support End')],
                neckline=resistance
            ))

        elif abs(low_slope) < 0.002 and high_slope < -0.001:
            support = np.mean(lows[-10:])
            status = 'completed' if current_price < support else 'forming'
            confidence = 0.85 if status == 'completed' else 0.6
            target = support - (max(highs) - support)
            patterns.append(PatternSignal(
                name="Descending Triangle", type="continuation", status=status, direction="bearish",
                confidence=confidence, quality_score=self._calculate_quality_score(status, confidence, 'bearish'),
                entry_zone=(support * 0.99, support * 1.01), stop_loss=max(highs[-10:]) * 1.02,
                targets=[target], strategy="Breakdown ribassista atteso sotto il supporto",
                timeframe="Swing (settimane)", description=f"Triangolo discendente. Supporto ${support:.2f}",
                warnings=["Attendere conferma volume al breakdown"],
                pattern_points=[(len(self.df) - lookback, support, 'S', 'Support'),
                                (len(self.df) - 1, high_line[-1], 'R', 'Resistance End')],
                neckline=support
            ))
        return patterns

    def detect_flag_pennant(self) -> List[PatternSignal]:
        patterns = []
        if len(self.df) < 15:
            return patterns
        recent = self.df.iloc[-20:] if len(self.df) >= 20 else self.df
        first_half = recent.iloc[:len(recent)//2]
        second_half = recent.iloc[len(recent)//2:]
        pole_return = (first_half['Close'].iloc[-1] - first_half['Close'].iloc[0]) / first_half['Close'].iloc[0]

        if abs(pole_return) > 0.05:
            highs = second_half['High'].values
            lows = second_half['Low'].values
            x = np.arange(len(highs))
            if len(highs) < 5:
                return patterns
            high_coef = np.polyfit(x, highs, 1)
            low_coef = np.polyfit(x, lows, 1)
            current_price = self.df['Close'].iloc[-1]

            if pole_return > 0 and high_coef[0] < -0.001 and low_coef[0] < -0.001:
                resistance = high_coef[1] + high_coef[0] * len(highs)
                start_idx = len(self.df) - len(second_half)
                status = 'completed' if current_price > resistance else 'forming'
                confidence = 0.85 if status == 'completed' else 0.6
                target = current_price * (1 + abs(pole_return))
                patterns.append(PatternSignal(
                    name="Bull Flag", type="continuation", status=status, direction="bullish",
                    confidence=confidence, quality_score=self._calculate_quality_score(status, confidence, 'bullish'),
                    entry_zone=(resistance * 0.99, resistance * 1.01), stop_loss=min(lows) * 0.98,
                    targets=[target], strategy="Entrare al breakout sopra la flag. Target = lunghezza pennone",
                    timeframe="Daytrading/Swing", description=f"Bull Flag dopo rialzo {pole_return*100:.1f}%",
                    warnings=["Conferma volume al breakout"],
                    pattern_points=[(start_idx, highs[0], 'PF', 'Pole Start'), (start_idx + len(highs) - 1, highs[-1], 'FE', 'Flag End')]
                ))

            elif pole_return < 0 and high_coef[0] > 0.001 and low_coef[0] > 0.001:
                support = low_coef[1] + low_coef[0] * len(lows)
                start_idx = len(self.df) - len(second_half)
                status = 'completed' if current_price < support else 'forming'
                confidence = 0.85 if status == 'completed' else 0.6
                target = current_price * (1 + pole_return)
                patterns.append(PatternSignal(
                    name="Bear Flag", type="continuation", status=status, direction="bearish",
                    confidence=confidence, quality_score=self._calculate_quality_score(status, confidence, 'bearish'),
                    entry_zone=(support * 0.99, support * 1.01), stop_loss=max(highs) * 1.02,
                    targets=[target], strategy="Entrare al breakdown sotto la flag. Target = lunghezza pennone",
                    timeframe="Daytrading/Swing", description=f"Bear Flag dopo ribasso {abs(pole_return)*100:.1f}%",
                    warnings=["Conferma volume al breakdown"],
                    pattern_points=[(start_idx, lows[0], 'PF', 'Pole Start'), (start_idx + len(lows) - 1, lows[-1], 'FE', 'Flag End')]
                ))
        return patterns

    def detect_candlestick_patterns(self) -> List[PatternSignal]:
        patterns = []
        if len(self.df) < 3:
            return patterns
        c1 = self.df.iloc[-1]
        c2 = self.df.iloc[-2]
        c3 = self.df.iloc[-3] if len(self.df) >= 3 else None
        patterns.extend(self._detect_single_candles(c1, len(self.df)-1))
        patterns.extend(self._detect_two_candles(c1, c2, len(self.df)-1, len(self.df)-2))
        if c3 is not None:
            patterns.extend(self._detect_three_candles(c1, c2, c3, len(self.df)-1, len(self.df)-2, len(self.df)-3))
        return patterns

    def _detect_single_candles(self, c, idx) -> List[PatternSignal]:
        patterns = []
        open_, high, low, close = c['Open'], c['High'], c['Low'], c['Close']
        body = abs(close - open_)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        total_range = high - low
        if total_range == 0:
            return patterns

        if body <= total_range * 0.3 and lower_shadow >= body * 2 and upper_shadow <= body * 0.5:
            if len(self.df) >= 6:
                trend_5d = (close - self.df['Close'].iloc[-6]) / self.df['Close'].iloc[-6]
                if trend_5d < -0.03:
                    patterns.append(PatternSignal(
                        name="Hammer (Rialzista)", type="reversal", status="completed", direction="bullish",
                        confidence=0.7, quality_score=75,
                        entry_zone=(close*0.99, close*1.01), stop_loss=low*0.99, targets=[close*1.03],
                        strategy="Inversione rialzista. Entrare sopra il massimo del hammer",
                        timeframe="1-3 giorni", description="Hammer dopo discesa. Rifiuto prezzi bassi",
                        warnings=["Attendere conferma rialzista"],
                        pattern_points=[(idx, low, 'Hammer', 'Hammer')]
                    ))
                elif trend_5d > 0.03:
                    patterns.append(PatternSignal(
                        name="Hanging Man (Ribassista)", type="reversal", status="completed", direction="bearish",
                        confidence=0.65, quality_score=70,
                        entry_zone=(close*0.99, close*1.01), stop_loss=high*1.01, targets=[close*0.97],
                        strategy="Possibile inversione ribassista. Attendere conferma",
                        timeframe="1-3 giorni", description="Hanging Man dopo rialzo",
                        warnings=["Richiede forte conferma"],
                        pattern_points=[(idx, high, 'HangingMan', 'Hanging Man')]
                    ))

        if body <= total_range * 0.1:
            patterns.append(PatternSignal(
                name="Doji", type="bilateral", status="completed", direction="neutral",
                confidence=0.4, quality_score=50,
                entry_zone=(close*0.995, close*1.005), stop_loss=low*0.99, targets=[close*1.02],
                strategy="Indecisione. Attendere candela successiva",
                timeframe="1-2 giorni", description="Doji: mercato in equilibrio",
                warnings=["Non è un segnale operativo da solo"],
                pattern_points=[(idx, close, 'Doji', 'Doji')]
            ))

        if upper_shadow >= body * 2 and lower_shadow <= body * 0.3 and close < open_:
            patterns.append(PatternSignal(
                name="Shooting Star", type="reversal", status="completed", direction="bearish",
                confidence=0.65, quality_score=70,
                entry_zone=(close*0.99, close*1.01), stop_loss=high*1.01, targets=[close*0.98],
                strategy="Pattern ribassista. Entrare sotto il minimo",
                timeframe="1-3 giorni", description="Shooting Star. Rifiuto prezzi alti",
                warnings=["Conferma necessaria"],
                pattern_points=[(idx, high, 'ShootingStar', 'Shooting Star')]
            ))
        return patterns

    def _detect_two_candles(self, c1, c2, idx1, idx2) -> List[PatternSignal]:
        patterns = []
        o1, h1, l1, cl1 = c1['Open'], c1['High'], c1['Low'], c1['Close']
        o2, h2, l2, cl2 = c2['Open'], c2['High'], c2['Low'], c2['Close']
        body1 = abs(cl1 - o1)
        body2 = abs(cl2 - o2)

        if cl2 < o2 and cl1 > o1 and o1 <= cl2 and cl1 >= o2 and body1 > body2 * 1.2:
            patterns.append(PatternSignal(
                name="Bullish Engulfing", type="reversal", status="completed", direction="bullish",
                confidence=0.75, quality_score=82,
                entry_zone=(cl1*0.99, cl1*1.01), stop_loss=l1*0.98, targets=[h1*1.02],
                strategy="Forte segnale rialzista. Stop sotto minimo engulfing",
                timeframe="2-5 giorni", description="Candela rialzista ingloba precedente ribassista",
                warnings=["Migliore a livelli di supporto"],
                pattern_points=[(idx2, cl2, 'C1', 'Bear'), (idx1, cl1, 'C2', 'Bull')]
            ))

        if cl2 > o2 and cl1 < o1 and o1 >= cl2 and cl1 <= o2 and body1 > body2 * 1.2:
            patterns.append(PatternSignal(
                name="Bearish Engulfing", type="reversal", status="completed", direction="bearish",
                confidence=0.75, quality_score=82,
                entry_zone=(cl1*0.99, cl1*1.01), stop_loss=h1*1.02, targets=[l1*0.98],
                strategy="Forte segnale ribassista. Stop sopra massimo engulfing",
                timeframe="2-5 giorni", description="Candela ribassista ingloba precedente rialzista",
                warnings=["Migliore a livelli di resistenza"],
                pattern_points=[(idx2, cl2, 'C1', 'Bull'), (idx1, cl1, 'C2', 'Bear')]
            ))
        return patterns

    def _detect_three_candles(self, c1, c2, c3, idx1, idx2, idx3) -> List[PatternSignal]:
        patterns = []
        o1, cl1 = c1['Open'], c1['Close']
        o2, cl2 = c2['Open'], c2['Close']
        o3, cl3 = c3['Open'], c3['Close']

        if (cl3 < o3 and abs(o2 - cl2) <= abs(o3 - cl3) * 0.3 and cl1 > o1 and cl1 > (o3 + cl3) / 2):
            patterns.append(PatternSignal(
                name="Morning Star", type="reversal", status="completed", direction="bullish",
                confidence=0.8, quality_score=85,
                entry_zone=(cl1*0.99, cl1*1.01),
                stop_loss=min(c3['Low'], c2['Low'], c1['Low'])*0.98,
                targets=[c1['High']*1.03],
                strategy="Forte inversione rialzista. Stop sotto i minimi",
                timeframe="3-7 giorni", description="Tre candele: ribassista, indecisione, rialzista",
                warnings=["Pattern molto affidabile con volume"],
                pattern_points=[(idx3, cl3, 'C1', 'Bear'), (idx2, cl2, 'C2', 'Doji'), (idx1, cl1, 'C3', 'Bull')]
            ))

        if (cl3 > o3 and abs(o2 - cl2) <= abs(o3 - cl3) * 0.3 and cl1 < o1 and cl1 < (o3 + cl3) / 2):
            patterns.append(PatternSignal(
                name="Evening Star", type="reversal", status="completed", direction="bearish",
                confidence=0.8, quality_score=85,
                entry_zone=(cl1*0.99, cl1*1.01),
                stop_loss=max(c3['High'], c2['High'], c1['High'])*1.02,
                targets=[c1['Low']*0.97],
                strategy="Forte inversione ribassista. Stop sopra i massimi",
                timeframe="3-7 giorni", description="Tre candele: rialzista, indecisione, ribassista",
                warnings=["Pattern molto affidabile con volume"],
                pattern_points=[(idx3, cl3, 'C1', 'Bull'), (idx2, cl2, 'C2', 'Doji'), (idx1, cl1, 'C3', 'Bear')]
            ))
        return patterns

    def _calculate_quality_score(self, status, confidence, direction) -> int:
        base_score = int(confidence * 70)
        if status == 'completed':
            base_score += 20
        elif status == 'forming':
            base_score += 5
        if direction == 'bullish' and self.market_context.get('trend') == 'bullish':
            base_score += 10
        elif direction == 'bearish' and self.market_context.get('trend') == 'bearish':
            base_score += 10
        if self.market_context.get('volume') == 'high':
            base_score += 5
        return min(base_score, 100)

    def _generate_hns_strategy(self, status, pattern_type):
        if status == 'completed':
            return "ENTRATA: Dopo rottura neckline con volume. STOP: Sopra la spalla destra. TARGET: Altezza testa-neckline"
        return "PREPARAZIONE: Pattern in formazione. Attendere rottura neckline per entrare"

    def _generate_double_strategy(self, status, pattern_type):
        if status == 'completed':
            return "ENTRATA: Alla rottura del livello intermedio. STOP: Oltre il doppio massimo/minimo"
        return "MONITORAGGIO: Attendere rottura livello intermedio. Possibile rimbalzo dal secondo estremo"

    def _generate_warnings(self, status, pattern_type):
        w = []
        if status == 'forming':
            w.append("Pattern IN FORMAZIONE: potrebbe non completarsi")
        if pattern_type == 'reversal':
            w.append("Pattern di inversione: attendere conferma forte")
        w.append("Nessun pattern è infallibile. Gestire il rischio")
        return w

    def run_complete_analysis(self) -> List[PatternSignal]:
        all_patterns = []
        all_patterns.extend(self.detect_head_and_shoulders())
        all_patterns.extend(self.detect_double_top_bottom())
        all_patterns.extend(self.detect_triangle_patterns())
        all_patterns.extend(self.detect_flag_pennant())
        all_patterns.extend(self.detect_candlestick_patterns())
        all_patterns.sort(key=lambda x: x.quality_score, reverse=True)
        self.patterns_found = all_patterns
        return all_patterns

    def build_chart(self, ticker: str = "TICKER") -> plt.Figure:
        plot_periods = min(100, len(self.df))
        df_plot = self.df.iloc[-plot_periods:].copy()

        fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_price = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)

        for ax in [ax_price, ax_vol]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e')
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['top'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['right'].set_color('#30363d')
            ax.yaxis.label.set_color('#8b949e')
            ax.xaxis.label.set_color('#8b949e')
            ax.grid(True, alpha=0.15, color='#30363d')

        # Candlestick
        for i in range(len(df_plot)):
            color = '#26a641' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else '#f85149'
            date_num = mdates.date2num(df_plot.index[i])
            ax_price.plot([date_num, date_num],
                         [df_plot['Low'].iloc[i], df_plot['High'].iloc[i]],
                         color=color, linewidth=0.8, zorder=2)
            body_h = abs(df_plot['Close'].iloc[i] - df_plot['Open'].iloc[i])
            ax_price.add_patch(plt.Rectangle(
                (date_num - 0.35, min(df_plot['Open'].iloc[i], df_plot['Close'].iloc[i])),
                0.7, max(body_h, df_plot['Close'].iloc[i] * 0.001),
                color=color, alpha=0.9, zorder=3
            ))

        # SMA lines
        sma20 = self.df['Close'].rolling(20).mean().iloc[-plot_periods:]
        sma50 = self.df['Close'].rolling(50).mean().iloc[-plot_periods:]
        ax_price.plot(df_plot.index, sma20, color='#58a6ff', linewidth=1.2, alpha=0.7, label='SMA 20', zorder=4)
        ax_price.plot(df_plot.index, sma50, color='#f0883e', linewidth=1.2, alpha=0.7, label='SMA 50', zorder=4)

        # Pattern overlays
        colors_map = {'bullish': '#26a641', 'bearish': '#f85149', 'neutral': '#f0883e'}
        for pattern in self.patterns_found[:5]:
            color = colors_map.get(pattern.direction, '#ffffff')

            if pattern.pattern_points:
                valid = []
                for p in pattern.pattern_points:
                    if 0 <= p[0] < len(self.df):
                        valid.append((self.df.index[p[0]], p[1], p[2] if len(p) > 2 else ''))
                for date_v, price_v, label in valid:
                    ax_price.plot(date_v, price_v, 'o', color=color, markersize=10,
                                  markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                    if label:
                        ax_price.annotate(label, (date_v, price_v),
                                          xytext=(0, 14 if pattern.direction == 'bearish' else -14),
                                          textcoords='offset points', ha='center', fontsize=8,
                                          fontweight='bold', color=color, zorder=11,
                                          bbox=dict(boxstyle='round,pad=0.25', facecolor='#0d1117', alpha=0.8, edgecolor=color, linewidth=0.5))
                if len(valid) >= 2:
                    dates_v = [v[0] for v in valid]
                    prices_v = [v[1] for v in valid]
                    ax_price.plot(dates_v, prices_v, '--', color=color, linewidth=1.8, alpha=0.65, zorder=5)

            if pattern.neckline is not None:
                ax_price.axhline(y=pattern.neckline, color=color, linestyle='--', linewidth=1.3, alpha=0.75, zorder=6)

            ax_price.axhspan(pattern.entry_zone[0], pattern.entry_zone[1], alpha=0.1, color=color, zorder=1)

            for j, tgt in enumerate(pattern.targets):
                ax_price.axhline(y=tgt, color=color, linestyle=':', linewidth=1, alpha=0.55, zorder=6)
                ax_price.text(mdates.date2num(df_plot.index[-1]), tgt, f' T{j+1}',
                              color=color, fontsize=8, va='center', zorder=11)

            ax_price.axhline(y=pattern.stop_loss, color='#f0883e', linestyle='-.', linewidth=1, alpha=0.7, zorder=6)
            ax_price.text(mdates.date2num(df_plot.index[0]), pattern.stop_loss, ' SL',
                          color='#f0883e', fontsize=8, va='center', zorder=11)

        ax_price.set_title(f'{ticker}  —  Technical Pattern Analysis', fontsize=14,
                           fontweight='bold', color='#e6edf3', pad=12)
        ax_price.set_ylabel('Price ($)', color='#8b949e')
        ax_price.legend(loc='upper left', fontsize=8, facecolor='#161b22', edgecolor='#30363d',
                        labelcolor='#e6edf3')
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        # Volume
        vol_colors = ['#26a641' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i]
                      else '#f85149' for i in range(len(df_plot))]
        ax_vol.bar(df_plot.index, df_plot['Volume'], color=vol_colors, alpha=0.75, width=0.8)
        ax_vol.set_ylabel('Vol', color='#8b949e')
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)

        plt.tight_layout()
        return fig


# ============================================================
# STREAMLIT UI
# ============================================================

def direction_badge(direction):
    if direction == 'bullish':
        return '<span class="badge badge-green">🟢 RIALZISTA</span>'
    elif direction == 'bearish':
        return '<span class="badge badge-red">🔴 RIBASSISTA</span>'
    return '<span class="badge badge-orange">⚪ NEUTRO</span>'

def status_badge(status):
    if status == 'completed':
        return '<span class="badge badge-green">✅ COMPLETATO</span>'
    return '<span class="badge badge-orange">🔄 IN FORMAZIONE</span>'

def type_badge(ptype):
    labels = {'reversal': '🔁 Inversione', 'continuation': '➡️ Continuazione', 'bilateral': '↔️ Bilaterale'}
    return f'<span class="badge badge-blue">{labels.get(ptype, ptype)}</span>'

def score_color(score):
    if score >= 80:
        return '#26a641'
    elif score >= 60:
        return '#f0883e'
    return '#f85149'


def main():
    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configurazione")
        st.markdown("---")

        ticker_input = st.text_input("🔍 Ticker Symbol", value="AAPL",
                                     help="Es: AAPL, TSLA, MSFT, BTC-USD").upper().strip()

        period_options = {
            "1 Mese": "1mo", "3 Mesi": "3mo", "6 Mesi": "6mo",
            "1 Anno": "1y", "2 Anni": "2y"
        }
        period_label = st.selectbox("📅 Periodo", list(period_options.keys()), index=2)
        period = period_options[period_label]

        st.markdown("---")
        st.markdown("### 🎯 Filtri Pattern")
        show_classic = st.checkbox("Pattern Classici", value=True)
        show_candle = st.checkbox("Candlestick", value=True)

        st.markdown("---")
        run_btn = st.button("▶️ Avvia Analisi", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer</b><br>
        Questo strumento è esclusivamente educativo. I pattern tecnici non garantiscono risultati futuri.
        Il trading comporta rischi significativi.
        </div>
        """, unsafe_allow_html=True)

    # ── HEADER ───────────────────────────────────────────────
    st.markdown("# 📊 Pattern Recognition Pro")
    st.markdown("Riconoscimento automatico di pattern tecnici su dati di mercato in tempo reale")
    st.markdown("---")

    # ── INITIAL STATE ────────────────────────────────────────
    if not run_btn and 'recognizer' not in st.session_state:
        st.info("👈 Inserisci un ticker nella sidebar e premi **Avvia Analisi** per iniziare.")
        return

    # ── RUN ANALYSIS ─────────────────────────────────────────
    if run_btn or 'recognizer' in st.session_state:
        if run_btn:
            with st.spinner(f"📥 Download dati per **{ticker_input}**..."):
                df = yf.download(ticker_input, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty:
                    st.error(f"❌ Nessun dato trovato per **{ticker_input}**. Verifica il simbolo.")
                    return
                st.session_state['df'] = df
                st.session_state['ticker'] = ticker_input

            with st.spinner("🔍 Analisi pattern in corso..."):
                rec = AdvancedPatternRecognizer(df)
                patterns = rec.run_complete_analysis()

                # Filtra se necessario
                if not show_classic:
                    patterns = [p for p in patterns if p.type not in ['reversal', 'continuation', 'bilateral']
                                or p.name in ['Hammer (Rialzista)', 'Hanging Man (Ribassista)', 'Doji',
                                              'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing',
                                              'Morning Star', 'Evening Star']]
                if not show_candle:
                    candle_names = {'Hammer (Rialzista)', 'Hanging Man (Ribassista)', 'Doji',
                                    'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing',
                                    'Morning Star', 'Evening Star'}
                    patterns = [p for p in patterns if p.name not in candle_names]
                rec.patterns_found = patterns

                st.session_state['recognizer'] = rec
                st.session_state['patterns'] = patterns

        rec = st.session_state['recognizer']
        patterns = st.session_state['patterns']
        ctx = rec.market_context
        ticker = st.session_state['ticker']
        df = st.session_state['df']

        # ── MARKET CONTEXT METRICS ────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        trend_icon = "📈" if ctx.get('trend') == 'bullish' else "📉" if ctx.get('trend') == 'bearish' else "↔️"
        vol_color = "#f85149" if ctx.get('volatility') == 'high' else "#e6edf3"

        with col1:
            st.metric("💰 Prezzo", f"${ctx.get('current_price', 0):.2f}")
        with col2:
            trend_val = ctx.get('trend', 'N/A').upper()
            st.metric(f"{trend_icon} Trend", trend_val)
        with col3:
            st.metric("⚡ Volatilità", ctx.get('volatility', 'N/A').upper())
        with col4:
            st.metric("📦 Volume", ctx.get('volume', 'N/A').upper())
        with col5:
            st.metric("🎯 Pattern trovati", len(patterns))

        st.markdown("---")

        # ── CHART ─────────────────────────────────────────────
        if patterns:
            st.markdown("### 📈 Grafico con Pattern")
            fig = rec.build_chart(ticker)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("⚠️ Nessun pattern rilevato per questo ticker nel periodo selezionato.")
            st.markdown("### 📈 Grafico Prezzi")
            # Basic chart anyway
            fig2, ax = plt.subplots(figsize=(16, 6), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            plot_periods = min(100, len(df))
            df_plot = df.iloc[-plot_periods:]
            ax.plot(df_plot.index, df_plot['Close'], color='#58a6ff', linewidth=1.5)
            ax.set_title(f'{ticker} — Nessun pattern rilevato', color='#e6edf3')
            ax.tick_params(colors='#8b949e')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            ax.grid(True, alpha=0.15, color='#30363d')
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            return

        st.markdown("---")

        # ── PATTERN CARDS ─────────────────────────────────────
        st.markdown("### 🎯 Pattern Rilevati")

        tab_all, tab_bull, tab_bear, tab_neutral = st.tabs(
            ["🔍 Tutti", "🟢 Rialzisti", "🔴 Ribassisti", "⚪ Neutri"]
        )

        def render_patterns(pattern_list):
            if not pattern_list:
                st.info("Nessun pattern in questa categoria.")
                return

            for i, p in enumerate(pattern_list):
                border_class = p.direction if p.direction in ['bullish', 'bearish'] else 'neutral'
                color = '#26a641' if p.direction == 'bullish' else '#f85149' if p.direction == 'bearish' else '#f0883e'

                with st.container():
                    c_left, c_right = st.columns([3, 1])

                    with c_left:
                        st.markdown(f"""
                        <div class="pattern-card {border_class}">
                            <b style="font-size:15px;color:#e6edf3;">{p.name}</b><br>
                            {direction_badge(p.direction)} {status_badge(p.status)} {type_badge(p.type)}
                            <br><br>
                            <span style="color:#8b949e;font-size:12px;">{p.description}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with c_right:
                        score = p.quality_score
                        sc = score_color(score)
                        st.markdown(f"""
                        <div style="text-align:center;padding:20px 10px;">
                            <div style="font-size:36px;font-weight:bold;color:{sc};">{score}</div>
                            <div style="font-size:11px;color:#8b949e;">Quality Score</div>
                            <div style="font-size:11px;color:#8b949e;margin-top:4px;">{int(p.confidence*100)}% Confidenza</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander(f"📋 Dettagli — {p.name}"):
                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.markdown(f"**🎯 Entry Zone**")
                            st.code(f"${p.entry_zone[0]:.2f} — ${p.entry_zone[1]:.2f}")
                            st.markdown(f"**🛑 Stop Loss**")
                            st.code(f"${p.stop_loss:.2f}")
                        with d2:
                            st.markdown(f"**📌 Targets**")
                            for j, tgt in enumerate(p.targets):
                                st.code(f"T{j+1}: ${tgt:.2f}")
                        with d3:
                            st.markdown(f"**⏱️ Timeframe**")
                            st.write(p.timeframe)

                        st.markdown(f"**🎯 Strategia**")
                        st.info(p.strategy)

                        if p.warnings:
                            st.markdown("**⚠️ Warning**")
                            for w in p.warnings:
                                st.warning(w)

                    st.markdown("")

        with tab_all:
            render_patterns(patterns)
        with tab_bull:
            render_patterns([p for p in patterns if p.direction == 'bullish'])
        with tab_bear:
            render_patterns([p for p in patterns if p.direction == 'bearish'])
        with tab_neutral:
            render_patterns([p for p in patterns if p.direction == 'neutral'])

        # ── RAW DATA ──────────────────────────────────────────
        st.markdown("---")
        with st.expander("📄 Dati Grezzi (ultimi 30 giorni)"):
            st.dataframe(
                df.tail(30).style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close'])
                .background_gradient(subset=['Close'], cmap='RdYlGn'),
                use_container_width=True
            )

        # ── FOOTER ────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"""
        <div class="disclaimer">
        📊 <b>Pattern Recognition Pro</b> — Analisi effettuata il {datetime.now().strftime('%d/%m/%Y %H:%M')}
        per <b>{ticker}</b> | Periodo: <b>{period_label}</b> | Dati: Yahoo Finance<br><br>
        ⚠️ <b>DISCLAIMER:</b> Questo sistema è puramente educativo. I pattern tecnici non garantiscono risultati futuri.
        Il trading comporta rischi significativi di perdita del capitale. Consulta un consulente finanziario autorizzato prima di investire.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
