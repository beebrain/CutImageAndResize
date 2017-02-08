import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import operator
class Visualplot():
    def __init__(self):
        self.fig = plt.figure(1)
        self.fig2 = plt.figure(2)
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax2 = self.fig2.add_subplot(1, 1,1)
        self.line1, = self.ax1.plot([],[], 'r-')  # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax1.plot([], [], 'b-')  # Returns a tuple of line objects, thus the comma

        self.line3, = self.ax2.plot([], [], 'g-')  # Returns a tuple of line objects, thus the comma
        self.line4, = self.ax2.plot([], [], 'r-')  # Returns a tuple of line objects, thus the comma
        plt.ion()


    def plot(self,data):
        # self.ax.plot(np.append(self.line1.get_xdata(), dataX),np.append(self.line1.get_ydata(), dataY))
        print data['acc']
        x = np.arange(0,len(data['acc']))+1
        x=x.tolist()
        print x
        self.ax1.plot(x,data['acc'],'r-')
        self.ax1.plot(x,data['val_acc'],'b-')
        self.ax1.legend(['train', 'test'], loc='upper left')
        self.ax1.set_title('model accuracy')
        self.ax1.set_ylabel('accuracy')
        self.ax1.set_xlabel('epoch')
        index, value = max(enumerate(data['acc']), key=operator.itemgetter(1))
        self.ax1.annotate(value, xy=(index+1, value), xytext=(-40, -30),textcoords='offset points',arrowprops=dict(arrowstyle="->"))
        index, value = max(enumerate(data['val_acc']), key=operator.itemgetter(1))
        self.ax1.annotate(value, xy=(index + 1, value),xytext=(-40, -30),textcoords='offset points',arrowprops=dict(arrowstyle="->"))

        self.ax2.plot(x,data['loss'],'g-')
        self.ax2.plot(x,data['val_loss'],'r-')
        self.ax2.legend(['train', 'test'], loc='upper left')
        self.ax2.set_title('model loss')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epoch')
        index, value = min(enumerate(data['loss']), key=operator.itemgetter(1))
        self.ax2.annotate(value, xy=(index+1, value),xytext=(-50,20),textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"))
        index, value = min(enumerate(data['val_loss']), key=operator.itemgetter(1))
        self.ax2.annotate(value, xy=(index + 1, value), xytext=(-50, 20),textcoords='offset points',
                          arrowprops=dict(arrowstyle="->"))
        self.fig.canvas.draw()
        plt.pause(0.001)

    # hl, = plt.plot([], [])
    #
    # def update_line(hl, new_data):
    #     hl.set_xdata(np.append(hl.get_xdata(), new_data))
    #     hl.set_ydata(np.append(hl.get_ydata(), new_data))
    #     plt.draw()

ff = Visualplot()
#
acc = np.load("history.npy")
acc = acc.tolist()
print acc['acc']
for i in xrange(1000):
    ff.plot(acc)
