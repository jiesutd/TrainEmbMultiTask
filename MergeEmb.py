import sys

def mergeEmb(first_file, main_file, output_file):
	
	first_lines = open(first_file, 'r').readlines()
	main_lines = open(main_file, 'r').readlines()
	
	first_num = len(first_lines)
	main_num = len(main_lines)
	print "Merge embeding: ", first_file, ":", first_num, "+", main_file, ":", main_num, "to:", output_file
	first_dict = {}
	main_set = set()
	for idx in range(0, first_num):
		pair = first_lines[idx].split(" ", 1)
		first_dict[pair[0]] = pair[1]
	for idx in range(0, main_num):
		pair = main_lines[idx].split(" ", 1)
		main_set.add(pair[0])
	diff_keys = set(first_dict.keys()).difference(main_set)
	print "Added item num: ", len(diff_keys)

	out_file = open(output_file,'w')
	out_file.writelines(main_lines)

	for key in diff_keys:
		out_file.write(key + " " + first_dict[key])
	out_file.close()
	print "Merge embeding finished. ", output_file, " generated." 

if __name__ == '__main__':
	mergeEmb(sys.argv[1], sys.argv[2], sys.argv[3])




