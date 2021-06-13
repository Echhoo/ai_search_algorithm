import sys
import random
from enum import IntEnum
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QGridLayout, QMessageBox
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import Qt
import copy

solution = []
solutionStep = 0
times = 0


# 用枚举类表示方向
class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class NumberHuaRong(QWidget):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.zero_row = 0
        self.zero_column = 0
        self.gltMain = QGridLayout()
        self.initUI()

    def initUI(self):
        # 设置方块间隔
        self.gltMain.setSpacing(10)

        self.onInit()

        # 设置布局
        self.setLayout(self.gltMain)
        # 设置宽和高
        self.setFixedSize(400, 400)
        # 设置标题
        self.setWindowTitle('八数码问题')
        # 设置背景颜色
        self.setStyleSheet("background-color:gray;")
        self.show()

    # 初始化布局
    def onInit(self):
        # 产生顺序数组
        self.numbers = list(range(1, 9))
        self.numbers.append(0)

        # 将数字添加到二维数组
        for row in range(3):
            self.blocks.append([])
            for column in range(3):
                temp = self.numbers[row * 3 + column]

                if temp == 0:
                    self.zero_row = row
                    self.zero_column = column
                self.blocks[row].append(temp)
        print(self.blocks)
        # 打乱数组
        for i in range(500):
            random_num = random.randint(0, 3)
            self.move(Direction(random_num))

        self.updatePanel()

    # 检测按键
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up or key == Qt.Key_W:
            self.move(Direction.UP)
        if key == Qt.Key_Down or key == Qt.Key_S:
            self.move(Direction.DOWN)
        if key == Qt.Key_Left or key == Qt.Key_A:
            self.move(Direction.LEFT)
        if key == Qt.Key_Right or key == Qt.Key_D:
            self.move(Direction.RIGHT)
        if key == Qt.Key_Enter or key == Qt.Key_Space:
            global solution
            solutionLen = len(solution)
            global solutionStep
            global times
            self.blocks = solution[solutionLen - solutionStep - 3]
            print(self.blocks)
            solutionStep = solutionStep + 1
            times += 1
        self.updatePanel()
        if self.checkResult():
            if QMessageBox.Ok == QMessageBox.information(self, '挑战结果', '恭喜您完成挑战！总步数：' + str(times)):
                self.onInit()

    # 方块移动算法
    def move(self, direction):
        if (direction == Direction.UP):  # 上
            if self.zero_row != 2:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row + 1][self.zero_column]
                self.blocks[self.zero_row + 1][self.zero_column] = 0
                self.zero_row += 1
        if direction == Direction.DOWN:  # 下
            if self.zero_row != 0:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row - 1][self.zero_column]
                self.blocks[self.zero_row - 1][self.zero_column] = 0
                self.zero_row -= 1
        if direction == Direction.LEFT:  # 左
            if self.zero_column != 2:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row][self.zero_column + 1]
                self.blocks[self.zero_row][self.zero_column + 1] = 0
                self.zero_column += 1
        if direction == Direction.RIGHT:  # 右
            if self.zero_column != 0:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row][self.zero_column - 1]
                self.blocks[self.zero_row][self.zero_column - 1] = 0
                self.zero_column -= 1

    def updatePanel(self):
        for row in range(3):
            for column in range(3):
                self.gltMain.addWidget(Block(self.blocks[row][column]), row, column)

        self.setLayout(self.gltMain)

    # 检测是否完成
    def checkResult(self):
        # 先检测最右下角是否为0
        if self.blocks[2][2] != 0:
            return False

        for row in range(3):
            for column in range(3):
                # 运行到此处说名最右下角已经为0，pass即可
                if row == 2 and column == 2:
                    pass
                # 值是否对应
                elif self.blocks[row][column] != row * 3 + column + 1:
                    return False

        return True


class Block(QLabel):
    """ 数字方块 """

    def __init__(self, number):
        super().__init__()

        self.number = number
        self.setFixedSize(80, 80)

        # 设置字体
        font = QFont()
        font.setPointSize(30)
        font.setBold(True)
        self.setFont(font)

        # 设置字体颜色
        pa = QPalette()
        pa.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(pa)

        # 设置文字位置
        self.setAlignment(Qt.AlignCenter)

        # 设置背景颜色\圆角和文本内容
        if self.number == 0:
            self.setStyleSheet("background-color:white;border-radius:10px;")
        else:
            self.setStyleSheet("background-color:blue;border-radius:10px;")
            self.setText(str(self.number))


#########################################
app = QApplication(sys.argv)
ex = NumberHuaRong()
start = ex.blocks
start.append(0)
start.append(0)
target = [[1, 2, 3], [4, 5, 6], [7, 8, 0], -1]


def evaluate(state):
    global target
    f = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != target[i][j]:
                f += 1
    return f


def findSpace(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


def expand(state, t):
    x = findSpace(state)[0]
    y = findSpace(state)[-1]
    expState = []
    # up
    if x - 1 >= 0 and t == 0:
        up = copy.deepcopy(state)
        up[x - 1][y], up[x][y] = up[x][y], up[x - 1][y]
        expState.append(up)
        return up
    if x + 1 <= 2 and t == 1:
        down = copy.deepcopy(state)
        down[x + 1][y], down[x][y] = down[x][y], down[x + 1][y]
        expState.append(down)
        return down
    if y - 1 >= 0 and t == 2:
        left = copy.deepcopy(state)
        left[x][y - 1], left[x][y] = left[x][y], left[x][y - 1]
        expState.append(left)
        return left
    if y + 1 <= 2 and t == 3:
        right = copy.deepcopy(state)
        right[x][y + 1], right[x][y] = right[x][y], right[x][y + 1]
        expState.append(right)
        return right


def findBestID():
    global openValue
    tmin = 10
    flag = 0
    for i in range(len(openList)):
        if openList[i][-1] == 0 and openValue[i] < tmin:
            tmin = openValue[i]
            flag = i
    return flag


def judgeSame(state1, state2):
    for i in range(3):
        for j in range(3):
            if state1[i][j] != state2[i][j]:
                return False
    return True


def evaOpenValue():
    global openList
    global openValue
    openValue = []
    for i in range(len(openList)):
        temp = evaluate(openList[i])
        openValue.append(temp)


def evaCloseValue():
    global closeList
    global closeValue
    closeValue = []
    for i in range(len(closeList)):
        temp = evaluate(closeList[i])
        closeValue.append(temp)


openList = []
openValue = []
closeList = []
closeValue = []
openList.append(start)

tarID = 0
while openList is not None:
    evaOpenValue()
    minID = findBestID()
    if judgeSame(openList[minID], target):
        print('ok')
        tarID = minID
        break
    for t in range(4):
        expTemp = expand(openList[minID], t)
        if expTemp is not None:
            evai = evaluate(expTemp)

            expTemp[3] = minID
            expTemp[-1] = 0
            flag1 = 0
            for j in range(len(openList)):
                if judgeSame(expTemp, openList[j]) and openList[j][-1] == 0:
                    flag1 = 1
                    break
            flag2 = 0
            for j in range(len(closeList)):
                if judgeSame(expTemp, closeList[j]):
                    flag2 = 1
                    break
            if flag2 == 0 and flag1 == 0:
                openList.append(expTemp)
                openValue.append(evai)
    closeList.append(openList[minID])
    openList[minID][-1] = 1

temp = openList[tarID]

while temp[-2] != 0:
    for i in range(3):
        for j in range(3):
            print(temp[i][j], end=' ')
        print('\n')
    print('---------')
    solution.append(temp[:3])
    temp = openList[temp[-2]]

solution.append(temp[:3])
print(temp[:3])
solution.append(start[:3])
print(start[:3])
solutionStep = len(solution) - 1
print(solution)
sys.exit(app.exec_())
