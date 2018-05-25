import argparse

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--threshold', type=int, default='500000', \
        help='threshold for guess count (default: 500000)')
        parser.add_argument('--input_file', type=str, default='testguesscnt.txt', \
        help='file that contains guess counts of test data')
        args = parser.parse_args()
        return args

def main():
	args = parse_args()
	cnt = 0
	totalcnt = 0
	with open(args.input_file, 'r') as f:
		for line in f:
			totalcnt += 1
			if int(line) <= args.threshold:
				cnt += 1
	print("Guessed percentage = {}/{}".format(cnt, totalcnt))

if __name__ == '__main__':
        main()

