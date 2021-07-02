import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False #显示负号

def newline(p1,p2,color=None,maker = None):
    ax = plt.gca()
    xmin,xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin,ymax = ax.get_ybound()
    else:
        ymax = p1[1] +(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1] + (p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax],[ymin,ymax],color=color,marker=maker)
    ax.add_line(l)
    return l

if __name__ == '__main__':
    male = [1,1]
    female = [0,1]

    fig,ax = plt.subplots()

    m = ax.scatter(male[0],male[1],s = 60,c = 'blue',marker='x')
    ax.annotate("沈雁冰",male)

    f = ax.scatter(female[0],female[1],s=60,c='red',marker='o')
    ax.annotate("冰心",female)

    ax.legend((m,f),('男','女'))

    plt.xlim(-0.1,1.5)
    plt.ylim(-0.1,1.5)
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.title('性别分类问题')
    newline([0.5,0],[1,1.5])
    ax.annotate('3x-y-1.5=0',[0.75,0.6])

    plt.xlabel('特征1：是否含"雁"')
    plt.ylabel('特征2：是否含"冰"')

    plt.show()
