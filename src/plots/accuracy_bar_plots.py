import matplotlib.pyplot as plt
import numpy as np

architectures = ['Logistic Regression', 'Convolutional Network', 'Issa et al. Network', 'Transformer', 'ResNet-50', 'DenseNet-201', 'Stitched ResNet-50', 'Stitched DenseNet-201', 'MultiNet']
validation_accuracies = [51.38, 65.28, 73.61, 58.33, 80.56, 81.94, 81.94, 87.50, 84.72]
test_accuracies = [22.22, 13.19, 75.69, 15.28, 76.39, 73.61, 74.31, 79.17, 77.78]

'''validation_fig = plt.figure()
ax = validation_fig.add_axes([0.1,0.2,0.8,0.72])
ax.bar(architectures, validation_accuracies)
validation_fig.suptitle('Neural Architecture Validation Accuracies', fontsize=11)
plt.xlabel('Neural Architecture', fontsize=9)
plt.ylabel('Validation Accuracy', fontsize=9)
plt.xticks(rotation=55, fontsize=7)
validation_fig.savefig('validation_bar_plot.png')
plt.close(validation_fig)



test_fig = plt.figure()
ax = test_fig.add_axes([0.1,0.2,0.8,0.72])
ax.bar(architectures, test_accuracies)
test_fig.suptitle('Neural Architecture Test Accuracies', fontsize=11)
plt.xlabel('Neural Architecture', fontsize=9)
plt.ylabel('Test Accuracy', fontsize=9)
plt.xticks(rotation=55, fontsize=7)
test_fig.savefig('test_bar_plot.png')
plt.close(test_fig)'''



validation_test_fig = plt.figure()
ax = validation_test_fig.add_axes([0.1,0.32,0.8,0.6])
barWidth = 0.25
r1 = np.arange(len(validation_accuracies))
r2 = [x + barWidth for x in r1]
plt.bar(r1, validation_accuracies, color='#1111bf', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, test_accuracies, color='#44bf44', width=barWidth, edgecolor='white', label='var2')
plt.xlabel('Neural Architecture', fontsize=9)
plt.ylabel('Accuracy', fontsize=9)
plt.xticks(rotation=70, fontsize=7)
plt.title('Neural Architecture Validation and Test Accuracies', fontsize=15)
plt.xticks([r + barWidth for r in range(len(validation_accuracies))], architectures)
plt.legend(('Validation', 'Test'))
plt.axhline(y=75.69, color='r', linestyle='dotted')
validation_test_fig.savefig('val_test_bar_plot.png')
plt.close(validation_test_fig)