

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import numpy as np

from matplotlib.dates import DateFormatter

import matplotlib.dates as mdates

def create_enhanced_prediction_chart(ticker, dates, actual, predictions, std_rescaled, 

                                     df_processed, save_path, dpi=120):

    fig = plt.figure(figsize=(16, 12))

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.5)  

    colors = {

        'actual': '#2E86AB',

        'predicted': '#A23B72',

        'confidence': '#E8B4D9',

        'rsi_line': '#F18F01',

        'rsi_over': '#E63946',

        'rsi_under': '#06A77D',

        'macd': '#457B9D',

        'signal': '#E63946',

        'volume': '#6C757D'

    }

    ax1 = plt.subplot(gs[0])

    ax1.plot(dates, actual, label='Actual Prices', linewidth=2.5, 

             color=colors['actual'], alpha=0.9, zorder=3)

    ax1.plot(dates, predictions, label='Predicted Prices', linewidth=2.5, 

             color=colors['predicted'], linestyle='--', alpha=0.9, zorder=3)

    if std_rescaled is not None:

        ax1.fill_between(dates, 

                        predictions - std_rescaled, 

                        predictions + std_rescaled,

                        alpha=0.25, color=colors['confidence'], 

                        label='Confidence Interval', zorder=1)

    ax1.set_title(f'{ticker} Stock Price Prediction - Enhanced Analysis', 

                  fontsize=18, fontweight='bold', pad=15)

    ax1.set_ylabel('Price ($)', fontsize=13, fontweight='bold')

    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax1.set_facecolor('#F8F9FA')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.tick_params(axis='both', labelsize=10)

    ax2 = plt.subplot(gs[1], sharex=ax1)

    if 'RSI' in df_processed.columns:

        rsi_data = df_processed['RSI'].values[-len(dates):]

        ax2.plot(dates, rsi_data, color=colors['rsi_line'], linewidth=2, label='RSI')

        ax2.axhline(y=70, color=colors['rsi_over'], linestyle='--', 

                   linewidth=1.5, alpha=0.7, label='Overbought (70)')

        ax2.axhline(y=30, color=colors['rsi_under'], linestyle='--', 

                   linewidth=1.5, alpha=0.7, label='Oversold (30)')

        ax2.fill_between(dates, 70, 100, alpha=0.1, color=colors['rsi_over'])

        ax2.fill_between(dates, 0, 30, alpha=0.1, color=colors['rsi_under'])

        ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')

        ax2.set_ylim(0, 100)

        ax2.legend(loc='upper left', fontsize=9)

        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        ax2.set_facecolor('#F8F9FA')

        ax2.tick_params(axis='both', labelsize=9)

    ax3 = plt.subplot(gs[2], sharex=ax1)

    if 'MACD' in df_processed.columns and 'MACD_Signal' in df_processed.columns:

        macd_data = df_processed['MACD'].values[-len(dates):]

        signal_data = df_processed['MACD_Signal'].values[-len(dates):]

        macd_hist = macd_data - signal_data

        ax3.plot(dates, macd_data, color=colors['macd'], linewidth=2, label='MACD')

        ax3.plot(dates, signal_data, color=colors['signal'], linewidth=2, 

                label='Signal', linestyle='--')

        colors_hist = [colors['rsi_under'] if x > 0 else colors['rsi_over'] for x in macd_hist]

        ax3.bar(dates, macd_hist, color=colors_hist, alpha=0.3, width=0.8)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        ax3.set_ylabel('MACD', fontsize=12, fontweight='bold')

        ax3.legend(loc='upper left', fontsize=9)

        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        ax3.set_facecolor('#F8F9FA')

        ax3.tick_params(axis='both', labelsize=9)

    ax4 = plt.subplot(gs[3], sharex=ax1)

    if 'Volume' in df_processed.columns:

        volume_data = df_processed['Volume'].values[-len(dates):]

        ax4.bar(dates, volume_data, color=colors['volume'], alpha=0.6, width=0.8)

        ax4.set_ylabel('Volume', fontsize=12, fontweight='bold')

        ax4.set_xlabel('Date', fontsize=13, fontweight='bold')

        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

        ax4.set_facecolor('#F8F9FA')

        ax4.tick_params(axis='both', labelsize=9)

        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.text(0.99, 0.01, 'AI-Powered Stock Prediction', 

            ha='right', va='bottom', fontsize=8, alpha=0.5, style='italic')

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    plt.close()

def create_enhanced_backtesting_chart(ticker, cumulative_stock, cumulative_strategy, 

                                      actual, predictions, save_path, dpi=120):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 

                                    gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(cumulative_stock, label='Buy and Hold Strategy', 

            linewidth=3, color='#2E86AB', alpha=0.9)

    ax1.plot(cumulative_strategy, label='AI Trading Strategy', 

            linewidth=3, color='#F18F01', alpha=0.9)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax1.fill_between(range(len(cumulative_stock)), cumulative_stock, 0, 

                    alpha=0.1, color='#2E86AB')

    ax1.fill_between(range(len(cumulative_strategy)), cumulative_strategy, 0, 

                    alpha=0.1, color='#F18F01')

    final_stock_return = cumulative_stock[-1] * 100

    final_strategy_return = cumulative_strategy[-1] * 100

    ax1.set_title(f'{ticker} Backtesting Results\n' + 

                 f'Buy & Hold: {final_stock_return:.1f}% | ' +

                 f'AI Strategy: {final_strategy_return:.1f}%', 

                 fontsize=16, fontweight='bold', pad=15)

    ax1.set_ylabel('Cumulative Returns', fontsize=13, fontweight='bold')

    ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)

    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax1.set_facecolor('#F8F9FA')

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    prediction_errors = np.abs(actual - predictions)

    error_percentage = (prediction_errors / actual) * 100

    ax2.plot(error_percentage, color='#E63946', linewidth=2, alpha=0.7)

    ax2.fill_between(range(len(error_percentage)), error_percentage, 0, 

                    alpha=0.2, color='#E63946')

    avg_error = np.mean(error_percentage)

    ax2.axhline(y=avg_error, color='#457B9D', linestyle='--', 

               linewidth=2, label=f'Average Error: {avg_error:.1f}%')

    ax2.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')

    ax2.set_xlabel('Time Steps', fontsize=13, fontweight='bold')

    ax2.set_ylabel('Error (%)', fontsize=12, fontweight='bold')

    ax2.legend(loc='upper right', fontsize=11)

    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax2.set_facecolor('#F8F9FA')

    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    plt.close()

