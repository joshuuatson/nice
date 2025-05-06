import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, y =  make_classification(n_samples=1000, n_features=3, n_informative = 3, n_redundant = 0, n_classes=2, n_clusters_per_class = 1)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_data(X, y, title = "3D Data Visualization", elev=20, azim=30):
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection='3d')   #111 short for 1 row, 1 column, 1st subplot, projection tells it the plot should be 3d

    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
    #scatter then contains the metadata to access later
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.view_init(elev=elev, azim=azim)  # Set the viewing angle

    legend = ax.legend(*scatter.legend_elements(), title = "Classes")
    ax.add_artist(legend)
    #scatter.legend_elements() returns (handles, labels)
    #ax.legend(handles, labels) is how to add a legend to the plot
    #so ax.legend(*scatter.legend_elements()) is a shorthand for the above
    #the * unpacks the tuple into two arguments for the ax.legend function

    plt.show()

plot_3d_data(X, y)


class NN_classification(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn = nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, 15)
        self.layer3 = nn.Linear(15, output_dim)

        self.activation = activation_fn
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)

        return x

model_relu = NN_classification(input_dim=3, output_dim=2, activation_fn = nn.ReLU())
model_sigmoid = NN_classification(input_dim=3, output_dim=2, activation_fn = nn.Sigmoid())
model_tanh = NN_classification(input_dim=3, output_dim=2, activation_fn = nn.Tanh())

epochs = 300
criterion = nn.CrossEntropyLoss()
def train(model, X_train, y_train):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)  #the outpute of classification are logits
        loss = criterion(outputs, y_train) # CrossEntropyLoss expects logits
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
    
    # plt.plot(loss_history)
    # plt.title(f'Loss History - {model.activation.__class__.__name__}')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

    return loss_history


loss_relu = train(model_relu, X_train, y_train)
loss_sigmoid = train(model_sigmoid, X_train, y_train)
loss_tanh = train(model_tanh, X_train, y_train)
            
import pyvista as pv
grid = pv.ImageData()

# Create 3D grid
x_mesh = np.linspace(-3, 3, 51)
y_mesh = np.linspace(-3, 3, 51)
z_mesh = np.linspace(-3, 3, 51)
xx, yy, zz = np.meshgrid(x_mesh, y_mesh, z_mesh, indexing='ij')  

# Flatten to (N, 3)
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

model_relu.eval()
with torch.no_grad():
    inputs = torch.tensor(grid_points, dtype=torch.float32)
    logits = model_relu(inputs)  # shape (N, 1) or (N, 2)

    #for binary classification, use sigmoid
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits).numpy()
    else:
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # prob for class 1

# Reshape to match grid shape
probs = probs.reshape(xx.shape)

#set dimensions, origin and spacing
grid.dimensions = np.array(probs.shape) 
grid.origin = (x_mesh[0], y_mesh[0], z_mesh[0])  # start of grid
spacing = (x_mesh[1] - x_mesh[0], y_mesh[1] - y_mesh[0], z_mesh[1] - z_mesh[0])
grid.spacing = spacing  # spacing between points

#add the probability field
grid.point_data["probability"] = probs.ravel(order="F")  # Fortran order for VTK - what does this mean

#extract isosurface at 0.5 (decision boundary)
contours = grid.contour([0.5])

#plotting
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.6, label="Decision Boundary")
plotter.add_axes()

plotter.add_points(X, render_points_as_spheres=True, point_size=20,
                   scalars=y, cmap='viridis', label="Data")
plotter.show()