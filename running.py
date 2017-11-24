from subprocess import call
import subprocess

def alpha_test(alpha):
    p1 = subprocess.Popen(['./evaluate', '-a', str(alpha)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['./check'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['./compare-with-human-evaluation'], stdin=p2.stdout,                      stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    output = p3.communicate()[0]
    #print(p3.communicate()[0])
    return output


def ngram_test(n):
    p1 = subprocess.Popen(['./evaluate', '-ng', str(n)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['./check'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['./compare-with-human-evaluation'], stdin=p2.stdout,                      stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    output = p3.communicate()[0]
    #print(p3.communicate()[0])
    return output


def ngram_we_test(n, n1, n2, n3, n4):
    p1 = subprocess.Popen(['./evaluate', '-ng', str(n)
                              , '-w1', str(n1), '-w2', str(n2), '-w3', str(n3), '-w4', str(n4)
                           ], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['./check'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['./compare-with-human-evaluation'], stdin=p2.stdout,                      stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    output = p3.communicate()[0]
    #print(p3.communicate()[0])
    return output


def bleu_smooth_test(n, smooth=0, epsilon=0.1, sm_alpha=0, smth_k=0):
    p1 = subprocess.Popen(['./evaluate', '-ng', str(n)
                              , '-smth', str(smooth), '-eps', str(epsilon), '-sm_al', str(sm_alpha),
                           '-sm_k', str(smth_k)
                           ], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['./check'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['./compare-with-human-evaluation'], stdin=p2.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    output = p3.communicate()[0]
    # print(p3.communicate()[0])
    return output


def frange(start, stop, step, locale):
    for i in xrange(start, stop, step):
        yield i/float(locale)  # to get flaot, divide with locale and return it
# convention to allow import of this file as a module
if __name__ == '__main__':

    #for i in frange(800, 1000, 1, 1000):
        #print 'Alpha:', i
        #print(alpha_test(i))
        #print('----------------------------\n')
    #for i in range(1,5):
        #print('ngram:', i)
        #print(ngram_test(i))
        #print('----------------------------\n')
    #print(ngram_we_test(4, 0.25, 0.25, 0.25, 0.25))
    #print(ngram_we_test(4, 1, 1, 1, 1))
    print('Smooth tests')
    for i in range(4,8):
        print('Smoothing tech: ', i)
        print(bleu_smooth_test(4, i, 0.1, 5, 5))
        print('\n')



"""
code for finding best alpha
al = []
with open('/home/junlinux/Desktop/CSCI544_Last/hw7/alpha_test', 'r') as rd:
    same = False
    all_set = False
    for line in rd:

        if not same and line.find('Alpha') != -1:
            #print(line.split(',')[1].replace(')', ''))
            a1 = line.split(':')[1].replace(')', '').replace('\n', '')
            same = True
        if same and line.find('Accuracy') != -1:
            a2 = line.split('=')[1].replace('\n', '')
            #print(line.split('=')[1])
            same = False
            all_set = True
        if all_set:
            al.append((a1, a2))
            all_set = False
al2 = sorted(al, key=lambda x: x[1], reverse=True)
#print(al)
print(al2[:10])

"""