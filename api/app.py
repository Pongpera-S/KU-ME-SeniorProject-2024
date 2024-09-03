from flask import Flask, request, render_template, redirect, url_for, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__,template_folder="../templates",static_folder="../static")

def plot_truss(x, y, lda, nf=None, Fx=None, Fy=None):
    plt.figure()

    # Plot truss elements
    for i in range(len(lda)):
        node1 = lda[i, 0] - 1
        node2 = lda[i, 1] - 1
        plt.plot([x[node1], x[node2]], [y[node1], y[node2]], 'bo-')

    # Plot node labels
    node_labels = {}  # Dictionary to store labels to avoid duplication
    for i in range(len(lda)):
        node1 = lda[i, 0] - 1
        node2 = lda[i, 1] - 1
        if node1 not in node_labels:
            plt.text(x[node1], y[node1], f'{node1 + 1}', fontsize=12, ha='right')
            node_labels[node1] = True
        if node2 not in node_labels:
            if i < len(lda) - 1:  # Labels to the right for all elements except the last one
                plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='right')
            else:  # Labels to the left for the last element
                plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='left')
            node_labels[node2] = True

    # Plot forces as arrows in front of the truss elements, labeled by Fx and Fy (if either is non-zero)
    if nf is not None and Fx is not None and Fy is not None:
        arrow_length = np.mean(y)  # Adjust the arrow length as needed
        for i in range(len(nf)):
            node_index = nf[i] - 1
            if Fx[i] != 0 or Fy[i] != 0:  # Check if either Fx or Fy is non-zero
                force_mag = np.sqrt(Fx[i]**2 + Fy[i]**2)  # Magnitude of the force vector
                if force_mag > 0:
                    scale_factor = arrow_length / force_mag
                    dx = Fx[i] * scale_factor
                    dy = Fy[i] * scale_factor
                    plt.arrow(x[node_index], y[node_index], dx, dy, head_width=(arrow_length / 10), head_length=(arrow_length / 10), fc='r', ec='r', zorder=10, alpha=0.5)
                    # Label the arrow with Fx and Fy values
                    label_text = f'{Fx[i]} ' if Fx[i] != 0 else ''
                    label_text += f'{Fy[i]}' if Fy[i] != 0 else ''
                    plt.text(x[node_index] + dx, y[node_index] + dy, label_text, fontsize=10, ha='center', va='center', color='r')

    plt.title('Truss Structure')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.axis('equal')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def calculate_deformation(e, n, x, y, lda, nf, Fx, Fy, E, A, nb):
    F = np.zeros(2 * n)
    for i in range(len(nf)):
        F[2 * nf[i] - 2] = Fx[i]
        F[2 * nf[i] - 1] = Fy[i]

    # Length of truss members (in)
    L = np.zeros(e)
    for i in range(e):
        L[i] = np.sqrt((x[lda[i, 1] - 1] - x[lda[i, 0] - 1]) ** 2 + (y[lda[i, 1] - 1] - y[lda[i, 0] - 1]) ** 2)
        
    # Orientation of truss members (degree)
    theta = np.zeros(e)
    for i in range(e):
        theta[i] = np.degrees(np.arctan2((y[lda[i, 1] - 1] - y[lda[i, 0] - 1]), (x[lda[i, 1] - 1] - x[lda[i, 0] - 1])))
        if theta[i] < 0:
            theta[i] += 180

    # Element stiffness matrices
    ke = np.zeros((4, 4, e))
    for i in range(e):
        k = E[i] * A[i] / L[i]
        ke[:, :, i] = k * np.array([[np.cos(np.radians(theta[i])) ** 2, np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])),
                                    -np.cos(np.radians(theta[i])) ** 2, -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i]))],
                                    [np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i])) ** 2,
                                    -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])) ** 2],
                                    [-np.cos(np.radians(theta[i])) ** 2, -np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])),
                                    np.cos(np.radians(theta[i])) ** 2, np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i]))],
                                    [-np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])) ** 2,
                                    np.sin(np.radians(theta[i])) * np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i])) ** 2]])

    # Assemble the global stiffness matrix
    K = np.zeros((2 * n, 2 * n))
    for i in range(e):
        K[2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2, 2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2] += ke[0:2, 0:2, i]
        K[2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2, 2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2] += ke[0:2, 2:4, i]
        K[2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2, 2 * (lda[i, 0] - 1):2 * (lda[i, 0] - 1) + 2] += ke[2:4, 0:2, i]
        K[2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2, 2 * (lda[i, 1] - 1):2 * (lda[i, 1] - 1) + 2] += ke[2:4, 2:4, i]

    # Apply displacement boundary condition
    KG = K.copy()

    for i in range(len(nb)):
        K[2 * nb[i] - 2, :] = 0
        K[2 * nb[i] - 2, 2 * nb[i] - 2] = 1
        K[2 * nb[i] - 1, :] = 0
        K[2 * nb[i] - 1, 2 * nb[i] - 1] = 1

    # Nodal solution
    U = np.linalg.solve(K, F)

    # Determine the middle index U
    middle_index_U = len(U) // 2

    # Split the matrix U into two parts
    Ux = U[:middle_index_U]
    Uy = U[middle_index_U:]

    # Calculate reaction at the supports
    R = np.dot(KG, U) - F

    # Determine the middle index R
    middle_index_R = len(R) // 2

    # Split the matrix R into two parts
    Rx = R[:middle_index_R]
    Ry = R[middle_index_R:]

    # Local displacement of each member
    u = np.zeros((4, 1, e))
    for i in range(e):
        T = np.array([[np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i])), 0, 0],
                      [np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i])), 0, 0],
                      [0, 0, np.cos(np.radians(theta[i])), -np.sin(np.radians(theta[i]))],
                      [0, 0, np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i]))]])
        u[:, :, i] = np.dot(np.linalg.inv(T), np.array([[U[2 * (lda[i, 0] - 1)], U[2 * (lda[i, 0] - 1) + 1], U[2 * (lda[i, 1] - 1)], U[2 * (lda[i, 1] - 1) + 1]]]).T)

    # Calculate the stress in each element
    sigma = np.zeros(e)
    for i in range(e):
        sigma[i] = E[i] * (u[2, 0, i] - u[0, 0, i]) / L[i]

    return U, Ux, Uy, R, Rx, Ry, sigma


def plot_deformed_truss(x, y, lda, U, sigma):
    plt.figure()
    
    # Plot the original and deformed truss
    for i in range(len(lda)):
        xu = [x[lda[i, 0] - 1], x[lda[i, 1] - 1]]
        yu = [y[lda[i, 0] - 1], y[lda[i, 1] - 1]]
        xd = [xu[0] + 200 * U[2 * (lda[i, 0] - 1)], xu[1] + 200 * U[2 * (lda[i, 1] - 1)]]
        yd = [yu[0] + 200 * U[2 * (lda[i, 0] - 1) + 1], yu[1] + 200 * U[2 * (lda[i, 1] - 1) + 1]]
        
        # Plot original truss
        plt.plot(xu, yu, '--r')
        plt.plot(xu, yu, 'ob')
        
        # Plot deformed truss
        plt.plot(xd, yd, '-k')

    # Plot node labels
    node_labels = {}
    for i in range(len(lda)):
        node1 = lda[i, 0] - 1
        node2 = lda[i, 1] - 1
        if node1 not in node_labels:
            plt.text(x[node1], y[node1], f'{node1 + 1}', fontsize=12, ha='right')
            node_labels[node1] = True
        if node2 not in node_labels:
            plt.text(x[node2], y[node2], f'{node2 + 1}', fontsize=12, ha='left' if i == len(lda) - 1 else 'right')
            node_labels[node2] = True

    # Find and annotate the highest and lowest stress values
    max_stress = np.max(sigma)
    min_stress = np.min(sigma)
    max_stress_element = np.argmax(sigma)
    min_stress_element = np.argmin(sigma)

    max_stress_coords = [(x[lda[max_stress_element, 0] - 1], y[lda[max_stress_element, 0] - 1]), 
                         (x[lda[max_stress_element, 1] - 1], y[lda[max_stress_element, 1] - 1])]
    
    min_stress_coords = [(x[lda[min_stress_element, 0] - 1], y[lda[min_stress_element, 0] - 1]), 
                         (x[lda[min_stress_element, 1] - 1], y[lda[min_stress_element, 1] - 1])]

    plt.text(np.mean([coord[0] for coord in max_stress_coords]), 
             np.mean([coord[1] for coord in max_stress_coords]), 
             f'Max Stress: {max_stress:.2f}', fontsize=9, color='black', bbox=dict(facecolor='cyan', alpha=0.5, edgecolor='none'))

    plt.text(np.mean([coord[0] for coord in min_stress_coords]), 
             np.mean([coord[1] for coord in min_stress_coords]), 
             f'Min Stress: {min_stress:.2f}', fontsize=9, color='black', bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

    plt.title('Deformed Truss Structure')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.axis('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    e = int(request.form['elements'])
    n = int(request.form['nodes'])
    x = list(map(float, request.form.getlist('x[]')))
    y = list(map(float, request.form.getlist('y[]')))
    lda = []
    for i in range(e):
        start_node = int(request.form[f'lda_start_{i+1}'])
        end_node = int(request.form[f'lda_end_{i+1}'])
        lda.append([start_node, end_node])
    lda = np.array(lda)
    nf = list(map(int, request.form.getlist('nf[]')))
    Fx = list(map(float, request.form.getlist('Fx[]')))
    Fy = list(map(float, request.form.getlist('Fy[]')))

    print(f"elements: {e}, nodes: {n}, x: {x}, y: {y}, lda: {lda}, nf: {nf}, Fx: {Fx}, Fy: {Fy}")

    image_base64 = plot_truss(x, y, lda, nf=nf, Fx=Fx, Fy=Fy)

    return render_template('result.html', image_base64=image_base64, e=e, n=n, x=x, y=y, lda=lda, nf=nf, Fx=Fx, Fy=Fy)


@app.route('/calculate', methods=['POST'])
def calculate():
    e = int(request.form['elements'])
    n = int(request.form['nodes'])
    x = list(map(float, request.form.getlist('x[]')))
    y = list(map(float, request.form.getlist('y[]')))
    lda = []
    for i in range(e):
        start_node = int(request.form[f'lda_start_{i+1}'])
        end_node = int(request.form[f'lda_end_{i+1}'])
        lda.append([start_node, end_node])
    lda = np.array(lda)
    nf = list(map(int, request.form.getlist('nf[]')))
    Fx = list(map(float, request.form.getlist('Fx[]')))
    Fy = list(map(float, request.form.getlist('Fy[]')))
    nb = list(map(int, request.form.getlist('nb[]')))

    # Check E selection and get appropriate values
    if request.form['E_selection'] == 'single':
        E = [float(request.form['E_single'])] * e  # Replicate single E value for all elements
    else:
        E = list(map(float, request.form.getlist('E[]')))

    # Check A selection and get appropriate values
    if request.form['A_selection'] == 'single':
        A = [float(request.form['A_single'])] * e  # Replicate single A value for all elements
    else:
        A = list(map(float, request.form.getlist('A[]')))

    print(f"elements: {e}, nodes: {n}, x: {x}, y: {y}, lda: {lda}, nf: {nf}, Fx: {Fx}, Fy: {Fy}, E: {E}, A: {A}, nb: {nb}")

    U, Ux, Uy, R, Rx, Ry, sigma = calculate_deformation(e, n, x, y, lda, nf, Fx, Fy, E, A, nb)
    deformed_image_base64 = plot_deformed_truss(x, y, lda, U, sigma)

    return render_template('deformed_result.html', deformed_image_base64=deformed_image_base64, U=U, R=R, Ux=Ux, Rx=Rx, Uy=Uy, Ry=Ry, sigma=sigma)


@app.route('/data')
def data():
    
    e = int(request.form['elements'])
    n = int(request.form['nodes'])
    x = list(map(float, request.form.getlist('x[]')))
    y = list(map(float, request.form.getlist('y[]')))
    lda = []
    for i in range(e):
        start_node = int(request.form[f'lda_start_{i+1}'])
        end_node = int(request.form[f'lda_end_{i+1}'])
        lda.append([start_node, end_node])
    lda = np.array(lda)
    nf = list(map(int, request.form.getlist('nf[]')))
    Fx = list(map(float, request.form.getlist('Fx[]')))
    Fy = list(map(float, request.form.getlist('Fy[]')))
    E = float(request.form['E'])
    A = float(request.form['A'])
    nb = list(map(int, request.form.getlist('nb[]')))

    Ux, Uy, Rx, Ry, sigma = calculate_deformation(e, n, x, y, lda, nf, Fx, Fy, E, A, nb)

    # Return JSON response
    return jsonify(Ux=Ux, Uy=Uy, Rx=Rx, Ry=Ry, sigma=sigma, lda=lda)


if __name__ == '__main__':
    app.run(debug=True)
