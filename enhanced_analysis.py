"""
Improved Crowd Management System - Enhanced Detection
-----------------------------------------------------

This version adds:
1. Configurable model sizes (n, s, m, l, x)
2. Adjustable confidence thresholds
3. Better handling of difficult videos
4. Enhanced tracking parameters
5. Fixed encoding issues

Use this when the basic version has poor accuracy.
"""

from crowd_management_system import CrowdManagementSystem


def analyze_video_enhanced(video_path, output_name="enhanced_output"):
    """
    Enhanced analysis with better detection for difficult videos

    Args:
        video_path: Path to your video file
        output_name: Name for output files (without extension)
    """

    print("\n" + "=" * 70)
    print("ENHANCED CROWD MANAGEMENT SYSTEM")
    print("=" * 70 + "\n")

    # Try different configurations based on video difficulty
    configs = [
        {
            'name': 'Standard (Fast)',
            'model_size': 'n',
            'confidence': 0.35,
            'description': 'Good for clear videos with good lighting'
        },
        {
            'name': 'Balanced',
            'model_size': 's',
            'confidence': 0.30,
            'description': 'Better detection, moderate speed'
        },
        {
            'name': 'High Accuracy',
            'model_size': 'm',
            'confidence': 0.25,
            'description': 'Best for difficult videos (slow motion, poor lighting)'
        }
    ]

    print("Available configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Model: YOLOv8-{config['model_size'].upper()}")
        print(f"   Confidence: {config['confidence']}")
        print(f"   Use case: {config['description']}")

    print("\n" + "-" * 70)
    choice = input("\nSelect configuration (1-3) or press Enter for auto-select: ").strip()

    if choice == '':
        print("\nAuto-selecting configuration based on video...")
        config = configs[1]  # Use balanced by default
    else:
        try:
            config = configs[int(choice) - 1]
        except:
            print("Invalid choice, using balanced configuration")
            config = configs[1]

    print(f"\n✓ Using: {config['name']}")
    print(f"  Model: YOLOv8-{config['model_size'].upper()}")
    print(f"  Confidence: {config['confidence']}\n")

    # Initialize with selected configuration
    system = CrowdManagementSystem(
        video_path=video_path,
        model_size=config['model_size'],
        confidence_threshold=config['confidence']
    )

    # Process video
    print("Processing video...\n")
    system.process_video(
        output_path=f'{output_name}.mp4',
        show_live=True
    )

    # Generate report with UTF-8 encoding
    system.generate_report(f'{output_name}_report.txt')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✓ Output video: {output_name}.mp4")
    print(f"✓ Report: {output_name}_report.txt")
    print(f"\nDetection Summary:")
    print(f"  Peak Count: {system.max_count_observed}")
    print(f"  Total Entries: {system.entry_count}")
    print(f"  Total Exits: {system.exit_count}")
    print(f"  Overcrowding: {'YES - ALERT TRIGGERED' if system.overcrowding else 'NO - SAFE'}")
    print("=" * 70 + "\n")


def batch_analyze_with_different_configs(video_path):
    """
    Run the same video with multiple configurations to compare results
    Helps you find the best settings for your specific video type
    """
    print("\n" + "=" * 70)
    print("BATCH ANALYSIS - Compare Different Configurations")
    print("=" * 70 + "\n")

    configs = [
        {'model': 'n', 'conf': 0.40, 'name': 'fast_strict'},
        {'model': 'n', 'conf': 0.30, 'name': 'fast_sensitive'},
        {'model': 's', 'conf': 0.30, 'name': 'balanced'},
        {'model': 'm', 'conf': 0.25, 'name': 'accurate'},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(configs)}: Model={config['model'].upper()}, Confidence={config['conf']}")
        print('=' * 70)

        try:
            system = CrowdManagementSystem(
                video_path=video_path,
                model_size=config['model'],
                confidence_threshold=config['conf']
            )

            output_name = f"test_{config['name']}"
            system.process_video(
                output_path=f'{output_name}.mp4',
                show_live=False  # No live display for batch
            )

            system.generate_report(f'{output_name}_report.txt')

            results.append({
                'config': f"{config['model'].upper()}-{config['conf']}",
                'peak_count': system.max_count_observed,
                'entries': system.entry_count,
                'exits': system.exit_count,
                'overcrowding': system.overcrowding
            })

            print(f"✓ Completed: Peak={system.max_count_observed}, Entries={system.entry_count}")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Configuration':<15} {'Peak Count':<12} {'Entries':<10} {'Exits':<10} {'Alert':<10}")
    print("-" * 70)

    for result in results:
        alert = "YES" if result['overcrowding'] else "NO"
        print(
            f"{result['config']:<15} {result['peak_count']:<12} {result['entries']:<10} {result['exits']:<10} {alert:<10}")

    print("\n" + "=" * 70)
    print("Review the output videos to see which configuration works best!")
    print("=" * 70 + "\n")


def quick_test_with_recommendations(video_path):
    """
    Quick test that gives recommendations for improving detection
    """
    print("\n" + "=" * 70)
    print("QUICK TEST & RECOMMENDATIONS")
    print("=" * 70 + "\n")

    print("Running initial test with standard settings...\n")

    # Test with standard settings
    system = CrowdManagementSystem(
        video_path=video_path,
        model_size='n',
        confidence_threshold=0.35
    )

    system.process_video(
        output_path='quick_test_output.mp4',
        show_live=True
    )

    # Analyze results and give recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70 + "\n")

    peak_count = system.max_count_observed
    threshold = system.threshold
    utilization = (peak_count / threshold * 100) if threshold > 0 else 0

    print(f"Detection Results:")
    print(f"  Peak Count: {peak_count}")
    print(f"  Threshold: {threshold}")
    print(f"  Utilization: {utilization:.1f}%\n")

    # Provide recommendations
    if peak_count < 5:
        print("⚠️ Very low detection count detected!")
        print("\nRecommendations:")
        print("  1. Lower confidence threshold:")
        print("     system = CrowdManagementSystem(video_path, confidence_threshold=0.25)")
        print("  2. Use better model:")
        print("     system = CrowdManagementSystem(video_path, model_size='m')")
        print("  3. Check video quality:")
        print("     - Is the video very dark?")
        print("     - Are people very small in frame?")
        print("     - Is the video resolution too low?")

    elif peak_count > threshold * 2:
        print("⚠️ Very high detection count - possible false positives!")
        print("\nRecommendations:")
        print("  1. Increase confidence threshold:")
        print("     system = CrowdManagementSystem(video_path, confidence_threshold=0.45)")
        print("  2. Check for:")
        print("     - Statues, mannequins being detected as people")
        print("     - Reflections in windows")
        print("     - Posters or images of people")

    else:
        print("✓ Detection seems reasonable!")
        if utilization < 50:
            print("\nOptional improvements:")
            print("  - Try model_size='s' or 'm' for better accuracy")
            print("  - Lower confidence to 0.30 to catch more people")
        else:
            print("\n✓ Current settings appear optimal for this video")

    print("\n" + "=" * 70 + "\n")


# ============================================================================
# MAIN USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import os

    print("\n" + "=" * 70)
    print("  ENHANCED CROWD MANAGEMENT SYSTEM")
    print("  Better Detection for Difficult Videos")
    print("=" * 70)

    # Get video path
    video_path = input("\nEnter video path (or press Enter for 'crowd_video.mp4'): ").strip()
    if not video_path:
        video_path = "crowd_video.mp4"

    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file '{video_path}' not found!")
        print("Please provide a valid video file path.\n")
        exit(1)

    print("\n" + "=" * 70)
    print("Choose Analysis Mode:")
    print("=" * 70)
    print("\n1. Enhanced Single Analysis (Recommended)")
    print("   - Choose configuration based on your video")
    print("   - Best for most users")

    print("\n2. Quick Test with Recommendations")
    print("   - Fast test to see what works")
    print("   - Get suggestions for improvement")

    print("\n3. Batch Analysis (Compare All)")
    print("   - Test multiple configurations")
    print("   - Find the best settings")
    print("   - Takes longer but comprehensive")

    print("\n" + "=" * 70)
    choice = input("\nSelect mode (1-3): ").strip()

    if choice == '1':
        analyze_video_enhanced(video_path)
    elif choice == '2':
        quick_test_with_recommendations(video_path)
    elif choice == '3':
        batch_analyze_with_different_configs(video_path)
    else:
        print("\nInvalid choice, running enhanced analysis...")
        analyze_video_enhanced(video_path)

    print("\n✅ All done! Check your output files.\n")