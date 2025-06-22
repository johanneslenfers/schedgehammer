rm taco_performance_distribution_results.json
touch taco_performance_distribution_results.json

for i in {1..500}; do
    python performance_distribution/taco_performance_distribution.py 50 $i
done