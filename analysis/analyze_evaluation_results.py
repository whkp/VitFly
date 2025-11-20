#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib # Add this if not already imported at the top
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EvaluationResultsAnalyzer:
    def __init__(self, results_dir="./evaluation_results", output_dir="./evaluation_analysis"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 配置matplotlib中文字体支持
        self.setup_chinese_fonts()
        
        # 模型配置
        self.model_configs = {
            'ViT': {'name': 'Vision Transformer', 'color': '#1f77b4', 'marker': 'o'},
            'ViTLSTM': {'name': 'ViT+LSTM', 'color': '#ff7f0e', 'marker': 's'},
            'ConvNet': {'name': 'ConvNet', 'color': '#2ca02c', 'marker': '^'},
            'LSTMNet': {'name': 'LSTMNet', 'color': '#9467bd', 'marker': 'x'},
            'RobustViTLSTM': {'name': 'Robust ViT+LSTM', 'color': '#d62728', 'marker': 'D'},
        }
        
        self.velocities = [3.0, 4.0, 5.0, 6.0, 7.0]
        
    def setup_chinese_fonts(self):
        """设置中文字体，解决matplotlib中文显示问题"""
        font_candidates = [
            'WenQuanYi Micro Hei',      # 推荐 Linux
            'Noto Sans CJK SC',         # 推荐 Linux / Mac
            'SimHei',                   # Windows
            'Microsoft YaHei',          # Windows
            'PingFang SC',              # Mac
            'Arial Unicode MS',         # 跨平台，但可能需要额外安装
            'DejaVu Sans',              # 广泛可用，但不一定完整支持中文
        ]
        found_font_name = None
        
        # 打印matplotlib的字体缓存路径，方便用户查找
        try:
            font_cache_path = matplotlib.get_cachedir()
            print(f"ℹ️ Matplotlib 字体缓存目录: {font_cache_path}")
            print(f"  如果中文显示持续有问题，请【务必手动删除】此目录下的所有 .json 文件 (例如 fontlist-v330.json) 或整个目录内容，然后重新运行脚本。")
        except Exception as e:
            print(f"ℹ️ 无法获取 Matplotlib 字体缓存目录: {e}")

        for font_name_candidate in font_candidates:
            try:
                # 尝试获取字体属性，如果字体族存在，这通常不会报错
                fm.FontProperties(family=font_name_candidate).get_name()
                # 进一步确认字体是否真的被系统识别 (这一步可能较慢，但更可靠)
                # font_path = fm.findfont(fm.FontProperties(family=font_name_candidate), fallback_to_default=False)
                # if font_path: # 确保路径存在
                found_font_name = font_name_candidate
                break 
            except RuntimeError: # Font family not found by FontProperties
                continue
            except Exception: # Other potential errors during font check
                continue
        
        if found_font_name:
            plt.rcParams['font.family'] = 'sans-serif' # 确保默认字体族是sans-serif
            plt.rcParams['font.sans-serif'] = [found_font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            print(f"✓ 中文字体尝试配置为: {found_font_name}. 请确保此字体已在您的系统上正确安装并包含中文字符集。")
        else:
            print("⚠ 未找到任何候选的中文字体。Matplotlib 可能无法正确显示中文。")
            print(f"  尝试的字体列表: {', '.join(font_candidates)}")
            print("  【重要】请执行以下操作:")
            print("  1. 确保您已安装至少一种支持中文的 TrueType 字体。")
            print("     对于 Linux (Debian/Ubuntu), 推荐安装 'WenQuanYi Micro Hei':")
            print("       sudo apt-get update && sudo apt-get install -y fonts-wqy-microhei")
            print("     对于其他系统，请搜索并安装相应的推荐中文字体。")
            print("  2. 安装或更改字体后，【必须手动删除】上面列出的 Matplotlib 字体缓存目录中的内容。")
            print("  3. 重新运行此 Python 脚本。")
            plt.rcParams['axes.unicode_minus'] = False # 即使中文失败，也尝试修复负号


    def load_evaluation_data(self):
        """加载所有评估数据"""
        all_data = []
        
        print("正在加载评估数据...")
        for model in self.model_configs.keys():
            for vel in self.velocities:
                folder_name = f"{model}_vel{vel}"
                yaml_file = self.results_dir / folder_name / f"evaluation_{model}_vel{vel}.yaml"
                
                if yaml_file.exists():
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        
                    if data:
                        for rollout_name, rollout_data in data.items():
                            record = {
                                'model': model,
                                'velocity': vel,
                                'rollout_name': rollout_name,
                                'success': rollout_data.get('Success', False),
                                'crashes': rollout_data.get('number_crashes', 0),
                                'finish_time': rollout_data.get('time_to_finish', np.nan),
                                'timestamp': rollout_data.get('timestamp', ''),
                                'segment_times': rollout_data.get('segment_times', {})
                            }
                            all_data.append(record)
                        print(f"✓ {model} @ {vel}m/s: {len(data)} 轮次")
                else:
                    print(f"✗ 未找到文件: {yaml_file}")
        
        self.df = pd.DataFrame(all_data)
        print(f"\n总共加载了 {len(self.df)} 条记录")
        return self.df
    
    def calculate_metrics(self):
        """计算各种性能指标"""
        metrics = []
        
        for model in self.model_configs.keys():
            for vel in self.velocities:
                subset = self.df[(self.df['model'] == model) & (self.df['velocity'] == vel)]
                
                if len(subset) > 0:
                    # 基本统计
                    total_runs = len(subset)
                    successful_runs = len(subset[subset['success'] == True])
                    success_rate = successful_runs / total_runs if total_runs > 0 else 0
                    
                    # 碰撞统计
                    total_crashes = subset['crashes'].sum()
                    avg_crashes = subset['crashes'].mean()
                    
                    # 时间统计
                    finish_times = subset['finish_time'].dropna()
                    avg_finish_time = finish_times.mean() if len(finish_times) > 0 else np.nan
                    std_finish_time = finish_times.std() if len(finish_times) > 0 else np.nan
                    
                    # 计算速度效率（理论时间 vs 实际时间）
                    theoretical_time = 60.0 / vel  # 60米距离 / 期望速度
                    time_efficiency = theoretical_time / avg_finish_time if not np.isnan(avg_finish_time) else 0
                    
                    metrics.append({
                        'model': model,
                        'velocity': vel,
                        'total_runs': total_runs,
                        'success_rate': success_rate,
                        'total_crashes': total_crashes,
                        'avg_crashes': avg_crashes,
                        'avg_finish_time': avg_finish_time,
                        'std_finish_time': std_finish_time,
                        'time_efficiency': time_efficiency,
                        'theoretical_time': theoretical_time
                    })
        
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df
    
    def plot_success_rates(self):
        """绘制成功率对比图"""
        plt.figure(figsize=(12, 8))
        
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                plt.plot(model_data['velocity'], model_data['success_rate'] * 100, 
                        marker=config['marker'], color=config['color'], 
                        label=config['name'], linewidth=2, markersize=8)
        
        plt.xlabel('速度 (m/s)', fontsize=12)
        plt.ylabel('成功率 (%)', fontsize=12)
        plt.title('不同模型在各速度下的成功率对比', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 成功率对比图已保存")
    
    def plot_crash_rates(self):
        """绘制碰撞率对比图"""
        plt.figure(figsize=(12, 8))
        
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                plt.plot(model_data['velocity'], model_data['avg_crashes'], 
                        marker=config['marker'], color=config['color'], 
                        label=config['name'], linewidth=2, markersize=8)
        
        plt.xlabel('速度 (m/s)', fontsize=12)
        plt.ylabel('平均碰撞次数', fontsize=12)
        plt.title('不同模型在各速度下的平均碰撞次数', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crash_rates_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 碰撞率对比图已保存")
    
    def plot_time_efficiency(self):
        """绘制时间效率对比图"""
        plt.figure(figsize=(12, 8))
        
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                plt.plot(model_data['velocity'], model_data['time_efficiency'], 
                        marker=config['marker'], color=config['color'], 
                        label=config['name'], linewidth=2, markersize=8)
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='理论最优效率')
        plt.xlabel('速度 (m/s)', fontsize=12)
        plt.ylabel('时间效率 (理论时间/实际时间)', fontsize=12)
        plt.title('不同模型在各速度下的时间效率对比', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 时间效率对比图已保存")
    
    def plot_performance_heatmap(self):
        """绘制性能热力图"""
        # 创建成功率热力图
        pivot_success = self.metrics_df.pivot(index='model', columns='velocity', values='success_rate')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_success, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': '成功率'}, vmin=0, vmax=1)
        plt.title('模型成功率热力图', fontsize=14, fontweight='bold')
        plt.xlabel('速度 (m/s)', fontsize=12)
        plt.ylabel('模型', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建碰撞次数热力图
        pivot_crashes = self.metrics_df.pivot(index='model', columns='velocity', values='avg_crashes')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_crashes, annot=True, fmt='.1f', cmap='Reds_r', 
                   cbar_kws={'label': '平均碰撞次数'})
        plt.title('模型平均碰撞次数热力图', fontsize=14, fontweight='bold')
        plt.xlabel('速度 (m/s)', fontsize=12)
        plt.ylabel('模型', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crash_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 性能热力图已保存")
    
    def plot_comprehensive_comparison(self):
        """绘制综合性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 成功率
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                axes[0,0].plot(model_data['velocity'], model_data['success_rate'] * 100, 
                             marker=config['marker'], color=config['color'], 
                             label=config['name'], linewidth=2)
        
        axes[0,0].set_xlabel('速度 (m/s)')
        axes[0,0].set_ylabel('成功率 (%)')
        axes[0,0].set_title('成功率对比', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 平均碰撞次数
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                axes[0,1].plot(model_data['velocity'], model_data['avg_crashes'], 
                             marker=config['marker'], color=config['color'], 
                             label=config['name'], linewidth=2)
        
        axes[0,1].set_xlabel('速度 (m/s)')
        axes[0,1].set_ylabel('平均碰撞次数')
        axes[0,1].set_title('碰撞率对比', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 平均完成时间
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                axes[1,0].plot(model_data['velocity'], model_data['avg_finish_time'], 
                             marker=config['marker'], color=config['color'], 
                             label=config['name'], linewidth=2)
        
        axes[1,0].set_xlabel('速度 (m/s)')
        axes[1,0].set_ylabel('平均完成时间 (秒)')
        axes[1,0].set_title('完成时间对比', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 时间效率
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                config = self.model_configs[model]
                axes[1,1].plot(model_data['velocity'], model_data['time_efficiency'], 
                             marker=config['marker'], color=config['color'], 
                             label=config['name'], linewidth=2)
        
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[1,1].set_xlabel('速度 (m/s)')
        axes[1,1].set_ylabel('时间效率')
        axes[1,1].set_title('时间效率对比', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('无人机避障模型综合性能对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 综合性能对比图已保存")
    
    def generate_summary_table(self):
        """生成性能汇总表"""
        # 计算总体性能指标
        summary_data = []
        
        for model in self.model_configs.keys():
            model_data = self.metrics_df[self.metrics_df['model'] == model]
            if len(model_data) > 0:
                overall_success = model_data['success_rate'].mean()
                overall_crashes = model_data['avg_crashes'].mean()
                overall_time_eff = model_data['time_efficiency'].mean()
                
                summary_data.append({
                    '模型': self.model_configs[model]['name'],
                    '平均成功率': f"{overall_success:.1%}",
                    '平均碰撞次数': f"{overall_crashes:.2f}",
                    '平均时间效率': f"{overall_time_eff:.3f}",
                    '总测试轮次': int(model_data['total_runs'].sum())
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        summary_df.to_csv(self.output_dir / 'performance_summary.csv', index=False, encoding='utf-8-sig')
        
        # 保存为表格图片
        fig, ax = plt.subplots(figsize=(12, len(summary_data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('模型性能汇总表', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 性能汇总表已保存")
        return summary_df
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report_lines = [
            "无人机避障模型评估分析报告",
            "=" * 50,
            f"分析时间: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
            f"数据来源: {self.results_dir}",
            f"分析轮次: {len(self.df)} 条记录",
            "",
            "测试配置:",
            f"- 模型数量: {len(self.model_configs)} 个",
            f"- 测试速度: {self.velocities} m/s",
            f"- 总配置数: {len(self.metrics_df)} 个",
            "",
            "关键发现:",
        ]
        
        # 分析最佳性能模型
        if len(self.metrics_df) > 0:
            best_success_model = self.metrics_df.loc[self.metrics_df['success_rate'].idxmax()]
            best_efficiency_model = self.metrics_df.loc[self.metrics_df['time_efficiency'].idxmax()]
            safest_model = self.metrics_df.loc[self.metrics_df['avg_crashes'].idxmin()]
            
            report_lines.extend([
                f"1. 最高成功率: {self.model_configs[best_success_model['model']]['name']} "
                f"在 {best_success_model['velocity']}m/s 下达到 {best_success_model['success_rate']:.1%}",
                f"2. 最高时间效率: {self.model_configs[best_efficiency_model['model']]['name']} "
                f"在 {best_efficiency_model['velocity']}m/s 下效率为 {best_efficiency_model['time_efficiency']:.3f}",
                f"3. 最安全模型: {self.model_configs[safest_model['model']]['name']} "
                f"在 {safest_model['velocity']}m/s 下平均碰撞 {safest_model['avg_crashes']:.2f} 次",
                "",
                "速度影响分析:",
            ])
            
            # 分析速度对性能的影响
            for vel in sorted(self.velocities):
                vel_data = self.metrics_df[self.metrics_df['velocity'] == vel]
                if len(vel_data) > 0:
                    avg_success = vel_data['success_rate'].mean()
                    avg_crashes = vel_data['avg_crashes'].mean()
                    report_lines.append(f"- {vel}m/s: 平均成功率 {avg_success:.1%}, 平均碰撞 {avg_crashes:.2f} 次")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        with open(self.output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✓ 分析报告已保存")
        return report_text
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始评估结果分析...")
        print("=" * 50)
        
        # 加载数据
        self.load_evaluation_data()
        
        # 计算指标
        self.calculate_metrics()
        
        # 生成图表
        self.plot_success_rates()
        self.plot_crash_rates()
        self.plot_time_efficiency()
        self.plot_performance_heatmap()
        self.plot_comprehensive_comparison()
        
        # 生成汇总和报告
        self.generate_summary_table()
        self.generate_analysis_report()
        
        print("\n" + "=" * 50)
        print("分析完成！")
        print(f"结果保存在: {self.output_dir}")
        print("生成的图表:")
        print("- success_rates_comparison.png (成功率对比)")
        print("- crash_rates_comparison.png (碰撞率对比)")
        print("- time_efficiency_comparison.png (时间效率对比)")
        print("- success_rate_heatmap.png (成功率热力图)")
        print("- crash_rate_heatmap.png (碰撞率热力图)")
        print("- comprehensive_comparison.png (综合对比)")
        print("- performance_summary_table.png (性能汇总表)")
        print("- performance_summary.csv (数据表格)")
        print("- analysis_report.txt (分析报告)")

if __name__ == "__main__":
    # 创建分析器并运行
    analyzer = EvaluationResultsAnalyzer(
        results_dir="./evaluation_results",
        output_dir="./evaluation_analysis"
    )
    
    analyzer.run_complete_analysis()