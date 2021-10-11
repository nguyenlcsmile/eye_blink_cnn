import tensorflow as tf 
import solver
import Blink_CNN
import load_data
import numpy as np 

tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

#Folder chứa datasets
path_data = 'data_20000.npy' 
data_dir = 'datasets/'
batch_size = 16
#Load_data
path_data = 'data_20000.npy'
#Load data train
data_annos = np.load(path_data, allow_pickle=True)
#Get data number
data_num = len(data_annos)
batch_num = np.int32(np.ceil(data_num/batch_size))

with tf.compat.v1.Session() as sess:
    #Build network 
    net = Blink_CNN.BlinkCNN(is_train=True)
    net.build()
    # print(net)

    #Init solver
    solver = solver.Solver(sess=sess, net=net)
    solver.init()
    # print(solver)

    print('Training...')
    summary_idx = 0
    for epoch in range(10):
        for i in range(batch_num):
            im_list, label_list, im_name_list = load_data.get_batch(data_annos, batch_num, 
                                                                    data_dir, i, batch_size,
                                                                    data_num, size=(224, 224))
            print(im_list.shape, label_list.shape)
            _, summary, prob, net_loss = solver.train(im_list, label_list)
            solver.writer.add_summary(summary, summary_idx)
            summary_idx += 1
            pred_label = np.argmax(prob, axis=-1)
            print('====================================')
            print('Net loss: {}'.format(net_loss))
            print('Real label: {}'.format(label_list))
            print('Pred label: {}'.format(pred_label))
            print('Epoch: {}'.format(epoch))
        if epoch % 5 == 0:
            solver.save(epoch)
