import sys

if len(sys.argv) != 3:
    print "Usage: $ python eval.py <results_file> <test_data_file>"
    exit()

results_file = open(sys.argv[1], "r")
results = results_file.read().splitlines()
results_file.close()
results_hash = dict.fromkeys(results)

test_data_file = open(sys.argv[2], "r")
test_data = test_data_file.read().splitlines()
test_data_file.close()

matched_lines = 0
total_lines = len(test_data)

for line in test_data:
    if results_hash.has_key(line):
        matched_lines += 1

print "matched_lines = " + str(matched_lines)
print "total_lines = " + str(total_lines)
guessed_percentage = float(matched_lines) / float(total_lines) * 100
print "guessed_percentage = " + str(guessed_percentage) + "%"

