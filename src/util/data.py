import csv


def transform(row):
	X = []

	# column Color
	if (row['Color']=='YELLOW'):
		X.append(0)
	else:
		X.append(1)

	# column Size
	if (row['Size']=='SMALL'):
		X.append(2)
	else:
		X.append(3)

	# column Act
	if (row['Act']=='STRETCH'):
		X.append(4)
	else:
		X.append(5)

	# column Age
	if (row['Age']=='ADULT'):
		X.append(6)
	else:
		X.append(7)

	y = 1 if row['Inflated']=='T' else 0
	#print 'y=%d' % (y)

	return (X, y)

# load and transform each row of a file (train/valid/test)
def generate(filename):
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)

		data = []
		for row in reader:
			X, y = transform(row)

			yield X, y
