#!/usr/bin/env python3
"""
Comparison Summary - Membandingkan hasil dari semua sistem deteksi
Menampilkan perbandingan performa dan hasil dari berbagai metode
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_detection_results():
    """Load semua hasil deteksi dari file JSON"""
    results = {}
    
    # Cari semua file JSON hasil deteksi
    json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'report' in f]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract method name dari filename
            method_name = json_file.replace('_report_', '').replace('.json', '')
            if method_name.startswith('enhanced_detection'):
                method_name = 'Enhanced Thin Line'
            elif method_name.startswith('ultra_sensitive'):
                method_name = 'Ultra Sensitive'
            elif method_name.startswith('petak_optimized'):
                method_name = 'Petak Optimizer'
            elif method_name.startswith('adaptive'):
                method_name = 'Adaptive Detector'
            elif method_name.startswith('robust'):
                method_name = 'Robust Detector'
            else:
                method_name = method_name.replace('_', ' ').title()
            
            results[method_name] = data
            
        except Exception as e:
            print(f"⚠️  Could not load {json_file}: {e}")
    
    return results

def create_comparison_visualization(results):
    """Buat visualisasi perbandingan"""
    if not results:
        print("❌ No detection results found for comparison")
        return
    
    # Prepare data
    methods = list(results.keys())
    petak_counts = [results[method]['total_petak'] for method in methods]
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart of petak counts
    bars = ax1.bar(methods, petak_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_title('Total Petak Detected by Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Petak')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, petak_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart of method distribution
    ax2.pie(petak_counts, labels=methods, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of Detected Petak', fontsize=14, fontweight='bold')
    
    # Area distribution comparison
    ax3.set_title('Area Distribution by Method', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Area (pixels)')
    ax3.set_ylabel('Count')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, (method, data) in enumerate(results.items()):
        if 'petak_details' in data and data['petak_details']:
            areas = [petak['properties']['area'] for petak in data['petak_details']]
            ax3.hist(areas, bins=10, alpha=0.6, label=method, color=colors[i % len(colors)])
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Aspect ratio comparison
    ax4.set_title('Aspect Ratio Distribution by Method', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Aspect Ratio')
    ax4.set_ylabel('Count')
    
    for i, (method, data) in enumerate(results.items()):
        if 'petak_details' in data and data['petak_details']:
            aspect_ratios = [petak['properties']['aspect_ratio'] for petak in data['petak_details']]
            ax4.hist(aspect_ratios, bins=10, alpha=0.6, label=method, color=colors[i % len(colors)])
    
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison
    output_path = 'detection_methods_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison visualization saved: {output_path}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Could not display comparison: {e}")
    
    return output_path

def print_detailed_comparison(results):
    """Print perbandingan detail"""
    print("\n" + "="*80)
    print("📊 DETAILED COMPARISON OF DETECTION METHODS")
    print("="*80)
    
    if not results:
        print("❌ No detection results found for comparison")
        return
    
    # Header
    print(f"{'Method':<20} {'Petak Count':<12} {'Avg Area':<12} {'Avg Aspect':<12} {'Best Method':<12}")
    print("-" * 80)
    
    best_method = None
    max_petak = 0
    
    for method, data in results.items():
        petak_count = data['total_petak']
        avg_area = 0
        avg_aspect = 0
        
        if 'petak_details' in data and data['petak_details']:
            areas = [petak['properties']['area'] for petak in data['petak_details']]
            aspect_ratios = [petak['properties']['aspect_ratio'] for petak in data['petak_details']]
            avg_area = np.mean(areas)
            avg_aspect = np.mean(aspect_ratios)
        
        if petak_count > max_petak:
            max_petak = petak_count
            best_method = method
        
        print(f"{method:<20} {petak_count:<12} {avg_area:<12.1f} {avg_aspect:<12.2f} {'':<12}")
    
    print("-" * 80)
    print(f"🏆 BEST METHOD: {best_method} ({max_petak} petak detected)")
    print("="*80)
    
    # Detailed analysis
    print("\n📈 DETAILED ANALYSIS:")
    print("-" * 50)
    
    for method, data in results.items():
        print(f"\n🔍 {method.upper()}:")
        print(f"   Total petak: {data['total_petak']}")
        
        if 'petak_details' in data and data['petak_details']:
            areas = [petak['properties']['area'] for petak in data['petak_details']]
            aspect_ratios = [petak['properties']['aspect_ratio'] for petak in data['petak_details']]
            rectangularities = [petak['properties']['rectangularity'] for petak in data['petak_details']]
            
            print(f"   Area range: {min(areas):.1f} - {max(areas):.1f} pixels")
            print(f"   Average area: {np.mean(areas):.1f} pixels")
            print(f"   Average aspect ratio: {np.mean(aspect_ratios):.2f}")
            print(f"   Average rectangularity: {np.mean(rectangularities):.3f}")
            
            # Size categories
            small_petak = sum(1 for area in areas if area < 1000)
            medium_petak = sum(1 for area in areas if 1000 <= area < 5000)
            large_petak = sum(1 for area in areas if area >= 5000)
            
            print(f"   Size distribution: Small({small_petak}) Medium({medium_petak}) Large({large_petak})")

def create_recommendation(results):
    """Buat rekomendasi berdasarkan hasil perbandingan"""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Find best method for different criteria
    best_total = max(results.items(), key=lambda x: x[1]['total_petak'])
    best_small_petak = None
    best_large_petak = None
    
    for method, data in results.items():
        if 'petak_details' in data and data['petak_details']:
            areas = [petak['properties']['area'] for petak in data['petak_details']]
            small_count = sum(1 for area in areas if area < 1000)
            large_count = sum(1 for area in areas if area >= 5000)
            
            if best_small_petak is None or small_count > best_small_petak[1]:
                best_small_petak = (method, small_count)
            if best_large_petak is None or large_count > best_large_petak[1]:
                best_large_petak = (method, large_count)
    
    print(f"🎯 BEST OVERALL: {best_total[0]} ({best_total[1]['total_petak']} petak)")
    
    if best_small_petak:
        print(f"🔍 BEST FOR SMALL PETAK: {best_small_petak[0]} ({best_small_petak[1]} small petak)")
    
    if best_large_petak:
        print(f"🏢 BEST FOR LARGE PETAK: {best_large_petak[0]} ({best_large_petak[1]} large petak)")
    
    print("\n📋 USAGE RECOMMENDATIONS:")
    print("   • For maximum petak detection: Use the method with highest total count")
    print("   • For small petak focus: Use method with most small petak")
    print("   • For large petak focus: Use method with most large petak")
    print("   • For balanced approach: Consider area distribution")
    
    print("\n🔧 TECHNICAL INSIGHTS:")
    print("   • Enhanced Thin Line: Best for preserving fine details")
    print("   • Ultra Sensitive: Best for maximum detection")
    print("   • Petak Optimizer: Best for balanced results")
    print("   • Adaptive Detector: Best for automatic parameter tuning")

def main():
    """Main function untuk comparison summary"""
    print("📊 DETECTION METHODS COMPARISON")
    print("=" * 60)
    print("Membandingkan hasil dari semua sistem deteksi")
    print("=" * 60)
    
    # Load results
    print("🔍 Loading detection results...")
    results = load_detection_results()
    
    if not results:
        print("❌ No detection results found")
        print("   Please run detection methods first to generate comparison data")
        return
    
    print(f"✅ Loaded {len(results)} detection methods:")
    for method in results.keys():
        print(f"   • {method}")
    
    # Create comparison visualization
    print(f"\n🎨 Creating comparison visualization...")
    comparison_path = create_comparison_visualization(results)
    
    # Print detailed comparison
    print_detailed_comparison(results)
    
    # Create recommendations
    create_recommendation(results)
    
    print(f"\n🎉 Comparison completed successfully!")
    print(f"📁 Comparison saved: {comparison_path}")
    print(f"\n💡 Next steps:")
    print(f"   • Review the comparison visualization")
    print(f"   • Choose the best method for your needs")
    print(f"   • Consider combining methods for optimal results")

if __name__ == "__main__":
    main()

