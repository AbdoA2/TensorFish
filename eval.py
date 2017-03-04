def evaluate(x1, x2, class_num):
    counts = [0] * class_num
    correct = [0] * class_num
    for i in range(len(x1)):
        counts[x1[i]] += 1
        if x1[i] == x2[i]:
            correct[x1[i]] += 1

    for i in range(class_num):
        print("Class %d: %.4f" % (i, correct[i]/counts[i]))

    print("Total: %.4f" % (sum(correct)/sum(counts)))