import heapq
import copy
import time
import math
import matplotlib.pyplot as plt

S0 = []
SG = []

# 上下左右四个方向移动
MOVE = {'up': [-1, 0],
        'down': [1, 0],
        'left': [0, -1],
        'right': [0, 1]}

# OPEN表
OPEN = []

# 节点的总数
SUM_NODE_NUM = 0


# 状态节点
class State(object):
    def __init__(self, depth=0, rest_dis=0.0, state=None, hash_value=None, father_node=None):
        """
        初始化
        :参数 depth: 从初始节点到目前节点所经过的步数
        :参数 rest_dis: 启发距离
        :参数 state: 节点存储的状态 4*4的列表
        :参数 hash_value: 哈希值，用于判重
        :参数 father_node: 父节点指针
        """
        self.depth = depth
        self.rest_dis = rest_dis
        self.fn = self.depth + self.rest_dis
        self.child = []  # 孩子节点
        self.father_node = father_node  # 父节点
        self.state = state  # 局面状态
        self.hash_value = hash_value  # 哈希值

    def __lt__(self, other):  # 用于堆的比较，返回距离最小的
        return self.fn < other.fn

    def __eq__(self, other):  # 相等的判断
        return self.hash_value == other.hash_value

    def __ne__(self, other):  # 不等的判断
        return not self.__eq__(other)


def cal_M_distence(cur_state):
    """
    计算曼哈顿距离
    :参数 state: 当前状态,4*4的列表, State.state
    :返回: M_cost 每一个节点计算后的曼哈顿距离总和
    """
    global num
    M_cost = 0
    # -15：
    for i in range(num):
        for j in range(num):
            # -8：
            # for i in range(3):
            #     for j in range(3):
            if cur_state[i][j] == SG[i][j]:
                continue
            number = cur_state[i][j]
            if number == 0:
                x, y = 3, 3
            else:
                x = number / 4  # 理论横坐标
                y = number - 4 * x - 1  # 理论的纵坐标
                M_cost += (abs(x - i) + abs(y - j))
    return M_cost


def cal_E_distence(cur_state):
    """
    计算曼哈顿距离
    :参数 state: 当前状态,4*4的列表, State.state
    :返回: M_cost 每一个节点计算后的曼哈顿距离总和
    """
    E_cost = 0
    for i in range(num):
        for j in range(num):
            if cur_state[i][j] == SG[i][j]:
                continue
            number = cur_state[i][j]
            if number == 0:
                x, y = 3, 3
            else:
                x = number / 4  # 理论横坐标
                y = number - 4 * x - 1  # 理论的纵坐标
                E_cost += math.sqrt((x - i) * (x - i) + (y - j) * (y - j))
    return E_cost


def generate_child(sn_node, sg_node, hash_set, open_table, cal_distence):
    """
    生成子节点函数
    :参数 sn_node:  当前节点
    :参数 sg_node:  最终状态节点
    :参数 hash_set:  哈希表，用于判重
    :参数 open_table: OPEN表
    :参数 cal_distence: 距离函数
    :返回: None
    """
    if sn_node == sg_node:
        heapq.heappush(open_table, sg_node)  # heappush(heap,item)建立大小堆
        print('已找到终止状态！')
        return
    global flag, num
    for i in range(0, num):
        for j in range(0, num):
            if sn_node.state[i][j] != 0:
                continue
            for d in ['left', 'up', 'down', 'right']:  # 四个偏移方向
                x = i + MOVE[d][0]
                y = j + MOVE[d][1]
                if x < 0 or x >= num or y < 0 or y >= num:  # 越界了
                    continue
                state = copy.deepcopy(sn_node.state)  # 复制父节点的状态
                state[i][j], state[x][y] = state[x][y], state[i][j]  # 交换位置
                h = hash(str(state))  # 哈希时要先转换成字符串

                if h in hash_set:  # 重复了
                    continue
                hash_set.add(h)  # 加入哈希表

                # 记录扩展节点的个数
                global SUM_NODE_NUM
                SUM_NODE_NUM += 1

                depth = sn_node.depth + 1  # 已经走的距离函数
                rest_dis = cal_distence(state)  # 启发的距离函数
                node = State(depth, rest_dis, state, h, sn_node)  # 新建节点
                sn_node.child.append(node)  # 加入到孩子队列
                heapq.heappush(open_table, node)  # 加入到堆中
                # show_block(state,depth)
                # print('child')

def plot_matrix(matrix, block, plt, zero_color=" ", another_color=" ", title=" ", step=" "):
    """
    plot_matrix: 用来画出矩阵；
    matrix为二维列表；
    plt为画笔，应该为：import matplotlib.pyplot as plt
    """
    plt.subplots(figsize=(4, 4))
    plt.title(title)
    plt.xlabel("Step " + step)
    rows = len(matrix)
    columns = len(matrix[0])
    #  -15：
    # plt.xlim(0, 4 * rows)
    # plt.ylim(0, 4 * columns)
    # -8:
    plt.xlim(0, num * rows)
    plt.ylim(0, num * columns)
    for i in range(rows):
        for j in range(columns):
            if flag == '8':
                if matrix[i][j] != 0:
                    # 画出一个3*3的矩形，其中左下角坐标为：(3 * j, 6 - 3 * i)，并填充颜色， 0和其他的要有区分；
                    plt.gca().add_patch(plt.Rectangle((3 * j, 6 - 3 * i), 3, 3, color=another_color, alpha=1))
                else:
                    plt.gca().add_patch(plt.Rectangle((3 * j, 6 - 3 * i), 3, 3, color=zero_color, alpha=1))
                plt.text(3 * j + 1.5, 7.5 - 3 * i, str(matrix[i][j]), fontsize=30, horizontalalignment='center')
            if flag == '15':
                if matrix[i][j] != 0:
                    # 画出一个4*4的矩形,并填充颜色， 0和其他的要有区分；
                    plt.gca().add_patch(plt.Rectangle((4 * j, 12 - 4 * i), 4, 4, color=another_color, alpha=1))
                else:
                    plt.gca().add_patch(plt.Rectangle((4 * j, 12 - 4 * i), 4, 4, color=zero_color, alpha=1))
                plt.text(4 * j + 2, 12.5 - 4 * i, str(matrix[i][j]), fontsize=30, horizontalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show(block=block)
    plt.pause(0.5)
    plt.close()


def show_block(block, step):
    print("------", step, "--------")
    for b in block:
        print(b)


def print_path(node):
    """
    输出路径
    :参数 node: 最终的节点
    :返回: None
    """
    print("最终搜索路径为：")
    steps = node.depth

    stack = []  # 模拟栈
    while node.father_node is not None:
        stack.append(node.state)  # 拓展节点
        node = node.father_node
    stack.append(node.state)
    step = 0
    while len(stack) != 0:
        t = stack.pop()  # 先入后出打印
        show_block(t, step)
        # 可视化
        plot_matrix(t, block=False, plt=plt, zero_color="#FFC050", another_color="#1D4946",
                    title="A* Algorithm", step=str(step))
        step += 1
    return steps  # 返回步数


def A_start(start, end, distance_fn, generate_child_fn):
    """
    A*算法
    :参数 start: 起始状态
    :参数 end: 终止状态
    :参数 distance_fn: 距离函数，可以使用自定义的
    :参数 generate_child_fn: 产生孩子节点的函数
    :返回: 最优路径长度
    """
    root = State(0, 0, start, hash(str(S0)), None)  # 根节点
    end_state = State(0, 0, end, hash(str(SG)), None)  # 最后的节点
    if root == end_state:
        print("start == end !")

    OPEN.append(root)
    heapq.heapify(OPEN)  # 成堆

    node_hash_set = set()  # 存储节点的哈希值
    node_hash_set.add(root.hash_value)
    while len(OPEN) != 0:
        top = heapq.heappop(OPEN)  # 返回最小值
        if top == end_state:  # 结束后直接输出路径
            return print_path(top)
        # 产生孩子节点，孩子节点加入OPEN表
        generate_child_fn(sn_node=top, sg_node=end_state, hash_set=node_hash_set,
                          open_table=OPEN, cal_distence=distance_fn)
    print("无搜索路径!")  # 没有路径
    return -1


if __name__ == '__main__':

    print('请输入数字：(八数码：8 十五数码：15)')

    flag = input()
    while True:
        if flag == '8':
            SG = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
            num = 3
            print('请输入初始八数码:')
            break;
        elif flag == '15':
            SG = [[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 0]]
            num = 4
            print('请输入初始15数码:')
            break;
        else:
            print('输入错误')
            flag = input()
    for i in range(num):
        S0.append(list(map(int, input().split())))

    print("选择距离计算方法(E:欧式距离计算启发函数，M:曼哈顿式距离计算启发函数")
    method = input()
    time1 = time.time()
    if method == 'E':
        length = A_start(S0, SG, cal_E_distence, generate_child)
    else:
        length = A_start(S0, SG, cal_M_distence, generate_child)
    time2 = time.time()
    if length != -1:
        if method == 'E':
            print("采用欧式距离计算启发函数")
        else:
            print("采用曼哈顿距离计算启发函数")
        print("搜索最优路径长度为", length)
        print("搜索时长为", (time2 - time1), "s")
        print("共检测节点数为", SUM_NODE_NUM)
