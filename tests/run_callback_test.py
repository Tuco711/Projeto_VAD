import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import update_dashboard_views

def run_test(selected_countries):
    try:
        outputs = update_dashboard_views(selected_countries, None, 'total_deaths')
        print('SUCCESS')
        print('Returned', len(outputs), 'outputs')
        for i, out in enumerate(outputs[:6]):
            print(i, type(out))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Test with a valid country list
    run_test(['Portugal', 'United States'])
    # Test with an invalid country to simulate user error
    run_test(['Nonexistentland', 'Narnia'])
