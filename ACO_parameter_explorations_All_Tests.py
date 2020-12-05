
### ==========================================================================================================================================
## ACO & CSO Comparison & ST Extensions
### ==========================================================================================================================================
# Authors: Henry Cerbone & Jennifer Shum
# CS286 Project
# ACO baseline developed from:  https://github.com/pjmattingly/ant-colony-optimization
# last updated: 12/04/2020

##note - to run existing sections, go to section 10 and uncomment out experiments of interest:
## Sections:---------------------------------------------------
# 1. Library Imports
# 2. Dataset imports
# 3. Define useful functions
# 4. Initial examples/using ACO method (simple how-to-use case)
# 5. Baseline results for 3 different nodesets
# 6. Extension 1: ACO Parameter Characterization and Analysis
# 7. Extension 2: CSO Test and Characterization
# 8. Extension 3: ACO vs CSO Comparison
# 9. Extension 4: Spatial-Temporal Behavior - ACO
# 10. Run Main (comment out which tests to run here)


### ==========================================================================================================================================
### 1. IMPORT LIBRARIES:
### ==========================================================================================================================================

# Import Alg. Code Bases from separate files:
from ant_colony_st import ant_colony_st #spatio-temporal version
from ant_colony import ant_colony
from cso_lib import CSO

#Import standard libraries for math/numbers/data/plotting:
import numpy as np
import math
import pickle
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

### ==========================================================================================================================================
### 2. Dataset Imports:
### ==========================================================================================================================================

#given some nodes, and some locations...
test_nodes = {0: (0, 7), 1: (3, 9), 2: (12, 4), 3: (14, 11), 4: (8, 11), 5: (15, 6), 6: (6, 15), 7: (15, 9), 8: (12, 10), 9: (10, 7)}

# a known benchmark node set:
oliver30_nodes = {0: (54,67), 1: (54,62), 2: (37, 84), 3: (41, 94), 4: (2, 99), 5: (7, 64), 6: (25, 62), 7: (22, 60), 8: (18, 54), 9: (4, 50), \
                  10: (13,40), 11: (18,40), 12: (24,42), 13: (25,38), 14: (44,35), 15: (41,26), 16: (45,21), 17: (58,35), 18: (62,32), 19: (82,7), \
                  20: (91,38), 21: (83,46), 22: (71,44), 23: (64,60), 24: (68,58), 25: (83,69), 26: (87,76), 27: (74,78), 28: (71,71), 29: (58,69)}


##Importing 2 other benchmark sets, 70 and 280 nodes respectively.
st_70_nodes = pickle.load( open( "st70.p", "rb" ) )
st_70_nodes[0] = st_70_nodes[70]
del st_70_nodes[70]

a_280_nodes = pickle.load( open( "a280.p", "rb" ) )
a_280_nodes[0] = a_280_nodes[280]
del a_280_nodes[280]

##Need to make sure relevant files were added in working directory:
#import os
#path = os.getcwd()
#print(path)


### ==========================================================================================================================================
### 3. Define Useful Functions
### ==========================================================================================================================================

#a function to get distance between nodes...
def distance(start, end):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    # c = sqrt(a^2 + b^2)
    return math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))



#Find Fitness (total distance from one node to the next in the path found) given a colony:
def get_fitness(colony, answer):
    #Find Fitness (total distance from one node to the next in the path found) given a colony:
    total_dist=0
    for i in range(len(answer)):
        if(i<len(answer)-1):
            node = answer[i]
            node2 = answer[i+1]
            total_dist+=colony.distance_matrix[node][node2]
        elif(i==len(answer)-1):
            node = answer[i]  #last
            node2 = answer[0] #first
            x=(colony.nodes[node][0])-(colony.nodes[node2][0])
            y=(colony.nodes[node][1])-(colony.nodes[node2][1])
            last_loop_term = abs(x) + abs(y)
            total_dist += last_loop_term
            #print(last_loop_term)
            #print(total_dist)
    return (total_dist)




#Function that sweeps a parameter with a given list of options and plots the performance of ACO:
#if "process_plot" = "True", will also plot the TSP X-Y plot of nodes over time.
def fitness_vs_param_sweep(param, param_test, list_distances, nodes=oliver30_nodes, process_plot = "False"):
    for j in param_test:
        #create based on parameter of interest:
        if(param == "iter"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, j)
            #colony = ant_colony_st(oliver30_nodes, distance, sun_travel, None, 50, 0.5, 1.2, 5, 0.4, 1000, j)
        elif(param =="ant"):
            colony = ant_colony(nodes, distance, None, j, 0.5, 1.2, 0.4, 1000, 80)
        elif(param == "alpha"):
            colony = ant_colony(nodes, distance, None, 50, j, 1.2, 0.4, 1000, 80)
        elif(param == "beta"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, j, 0.4, 1000, 80)
        elif(param == "evap"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, j, 1000, 80)
        elif(param == "evap2"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, 0.4, j, 80)
        else:
            (print("not a valid type of parameter!"))

        #run the ACO alg.
        answer = colony.mainloop()
        total_dist = get_fitness(colony, answer)

        ##Optional Output: print out the path and total distances found
        print(answer)
        print(total_dist)
        list_distances.append(total_dist)

        if(process_plot == True):
            # Plot Travelling Salesman Problem Nodes on X-Y Plot, and Distance:
            # https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp
            # plt.figure()
            x_list=[]
            y_list = []

            #Make a 2D array of answer node x and y locations
            answer.append(answer[0])
            for i in range(len(answer)):
                current_key = answer[i]
                #print(current_key)
                #nodes = test_nodes
                current_node = nodes[current_key]
                #print(current_node)
                x_list.append(current_node[0])
                y_list.append(current_node[1])
            positions = np.column_stack((x_list, y_list))
            #print(positions)

            #Create 2 subplots - one for raw nodes, one for colored answer orders and edges
            fig, ax = plt.subplots(2, sharex=True, sharey=True)                        # Prep 2 plots
            ax[0].set_title('Raw nodes')
            ax[1].set_title('ACO-ST Optimized Tour: Iter: ' + str(j))
            ax[0].scatter(positions[:, 0], positions[:, 1])                            # plot A
            s=ax[1].scatter(positions[:, 0], positions[:, 1],c=np.arange(len(x_list))) # plot B

            #colormap init:
            YlGnBl = plt.get_cmap('YlGnBu')
            YlGnBlr= YlGnBl.reversed()
            RdPu = plt.get_cmap('RdPu')

            #for all the nodes in answer, plot
            start_node = 0
            distance1 = 0.
            N=len(answer)
            for i in range(N-2):
                start_pos = positions[start_node]
                next_node = i+1
                #print(next_node)
                end_pos = positions[next_node]

                #Add arrows connecting edges of graph:
                # (switch scale based on num nodes - 8 (for Oliver30) to 25-50 for testnodes)
                ax[1].annotate("",
                        xy=start_pos, xycoords='data',
                        xytext=end_pos, textcoords='data',
                        arrowprops= dict(arrowstyle="->",
                                        connectionstyle="arc3",color=YlGnBlr(i*(8))))

                ##Unused: can also plot edges as color based upon pheromone map values:
                ##simply uncomment here to enable
                #pheromonemap = colony.pheromone_map
                #path_pheromone = pheromonemap[start_node][next_node]
                #print(path_pheromone)

                #s1=ax[1].annotate("",
                        #xy=start_pos, xycoords='data',
                        #xytext=end_pos, textcoords='data',
                        #arrowprops= dict(arrowstyle="->",
                                        #connectionstyle="arc3",color=YlGnBl(path_pheromone/20)))
                #distance1 += np.linalg.norm(end_pos - start_pos)
                #start_node = next_node

            # textbox overlay with distances:
            textstr = "N nodes: %d\nTotal length: %.3f" % (N-1, distance1)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10, # Textbox
                    verticalalignment='top', bbox=props)

            #cb = plt.colorbar(s)
            plt.tight_layout()
            plt.show()


    #Plot both our calculated ACO results against expected optimal performance for this benchmark
    #plt.plot(iter_test,list_distances,'-o')
    #ACO_results = plt.plot(param_test,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    #expected_results = plt.plot(param_test,[420] * len(param_test), label="Optimal", linestyle='--', color='g')
    return plt




### ==========================================================================================================================================
### 4. Initial Examples and Tests using ACO Method Baseline:
### ==========================================================================================================================================

def simple_example_ACO():
    #given some nodes, and some locations...
    print(test_nodes)

    #...we can make a colony of ants...
    colony = ant_colony(test_nodes, distance)
    #colony = ant_colony(test_nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, 10)

    #...to run ACO and get a tour
    answer = colony.mainloop()
    print(answer)
    total_dist = get_fitness(colony, answer)
    print(total_dist)



### ==========================================================================================================================================
### 5. Baseline results for 3 different nodesets
### ==========================================================================================================================================

##### 3 Benchmarks IterTests:--------------------------------------------------------------------------
def three_benchmark_iter(iter_test):
    ##iter_test = [1,2,4,10,50]
    #iter_test = [1,2,4,10]
    list_distances =[]
    plt.figure()

    fitness_vs_param_sweep("iter", iter_test, list_distances, oliver30_nodes)
    ACO_results = plt.plot(iter_test,list_distances, label="Oliver30 Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(iter_test,[423] * len(iter_test), label="Optimal O30", linestyle='--', color='b')

    list_distances =[]
    fitness_vs_param_sweep("iter", iter_test, list_distances, st_70_nodes)
    ACO_results = plt.plot(iter_test,list_distances, label="ST70 ACO Results", linestyle='-', marker='o',color='r')
    expected_results = plt.plot(iter_test,[675] * len(iter_test), label="Optimal ST70", linestyle='--', color='r')

    list_distances =[]
    fitness_vs_param_sweep("iter", iter_test, list_distances, a_280_nodes)
    ACO_results = plt.plot(iter_test,list_distances, label="A280 ACO Results", linestyle='-', marker='o',color='y')
    expected_results = plt.plot(iter_test,[2567] * len(iter_test), label="Optimal A280", linestyle='--', color='y')


    plt.title('Fitness (Path Distance) vs. Iterations')
    plt.xlabel('Num iterations')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()

#### 3 Benchmarks AntTests:----------------------------------------------------------------------------------

#ant_test = [1,2,3]

def three_benchmark_ant(ant_test):
    list_distances =[]
    plt.figure()

    fitness_vs_param_sweep("ant", ant_test, list_distances, oliver30_nodes)
    ACO_results = plt.plot(ant_test,list_distances, label="Oliver30 Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(ant_test,[423] * len(ant_test), label="Optimal O30", linestyle='--', color='b')

    list_distances =[]
    fitness_vs_param_sweep("ant", ant_test, list_distances, st_70_nodes)
    ACO_results = plt.plot(ant_test,list_distances, label="ST70 ACO Results", linestyle='-', marker='o',color='r')
    expected_results = plt.plot(ant_test,[675] * len(ant_test), label="Optimal ST70", linestyle='--', color='r')

    list_distances =[]
    fitness_vs_param_sweep("ant", ant_test, list_distances, a_280_nodes)
    ACO_results = plt.plot(ant_test,list_distances, label="A280 ACO Results", linestyle='-', marker='o',color='y')
    expected_results = plt.plot(ant_test,[2567] * len(ant_test), label="Optimal A280", linestyle='--', color='y')


    plt.title('Fitness (Path Distance) vs. Number Ants')
    plt.xlabel('Number Ants')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()


### ==========================================================================================================================================
### 6. Ext1: Single benchmark testing (oliver30) with parameter sweeps
### ==========================================================================================================================================

##### Test how the solution performs/converges with varying iterations:-------------------------------------------------------
def iter_test(iter_test_list):
    #iter_test_list = [1,2,4,10,50,200]
    #iter_test_list = [2, 5, 10]
    list_distances =[]
    plt.figure()

    fitness_vs_param_sweep("iter", iter_test_list, list_distances, oliver30_nodes)
    print(iter_test_list)
    print(list_distances)
    ACO_results = plt.plot(iter_test_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(iter_test_list,[420] * len(iter_test_list), label="Optimal", linestyle='--', color='g')

    plt.title('Fitness (Path Distance) vs. Iterations')
    plt.xlabel('Num iterations')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()



###### Test how the solution performs/converges with varying number of ants: --------------------------------------------------
def ant_test(ant_test_list):
    #ant_test_list = [1,2,4,10,50,100]
    list_distances =[]
    plt.figure()
    fitness_vs_param_sweep("ant", ant_test_list,list_distances)
    ACO_results = plt.plot(ant_test_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(ant_test_list,[420] * len(ant_test_list), label="Optimal", linestyle='--', color='g')

    plt.title('Fitness (Path Distance) vs. Number of Ants')
    plt.xlabel('Number Ants')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()


###### Test how the solution performs/converges with varying value for alpha: ==========================================================
def alpha_test(ant_test_list):
    ###alpha_test = [0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.99]
    #alpha_test_list = [0.1,0.25,0.5,0.75,1]
    list_distances =[]
    plt.figure()
    fitness_vs_param_sweep("alpha", alpha_test_list,list_distances)
    ACO_results = plt.plot(alpha_test_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(alpha_test_list,[420] * len(alpha_test_list), label="Optimal", linestyle='--', color='g')

    plt.title('Fitness (Path Distance) vs. Alpha Value')
    plt.xlabel('Alpha Value')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()



###### Test how the solution performs/converges with varying value for beta: ==========================================================
def beta_test(ant_test_list):
    #beta_test_list = [1, 1.2,2,3,4,5]
    list_distances =[]
    plt.figure()
    fitness_vs_param_sweep("beta", beta_test_list,list_distances)
    ACO_results = plt.plot(beta_test_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(beta_test_list,[420] * len(beta_test_list), label="Optimal", linestyle='--', color='g')

    plt.title('Fitness (Path Distance) vs. Beta Value')
    plt.xlabel('Beta Value')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()


###### Test how the solution performs/converges with varying alpha, beta:===============================================================
def alpha_beta_test(alpha_list, beta_list):
    list_distances =[]
    fig = plt.figure()
    alpha_plot_list = []
    beta_plot_list = []

    #Test ACO for each combination of alpha a and beta b:
    for a in alpha_list:
        for b in beta_list:
            #create ant colony with desired characteristics:
            nodes = oliver30_nodes
            colony = ant_colony(nodes, distance, None, 50, a, b, 0.4, 1000, 10)

            #solve with ACO:
            answer = colony.mainloop()
            print(answer)

            #get total distance/fitness
            total_dist = get_fitness(colony, answer)
            list_distances.append(total_dist)
            alpha_plot_list.append(a)
            beta_plot_list.append(b)

    #Arrange data into form for plot:
    shape = (len(alpha_list),len(beta_list))
    z_array = np.array(list_distances)
    z_array_2D= z_array.reshape(shape)
    #ACO_results = plt.plot(alpha_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')

    ax = plt.axes(projection='3d')
    p = ax.scatter3D(alpha_plot_list, beta_plot_list, list_distances, c=list_distances, cmap='RdPu');
    #ax.contour3D(alpha_list, beta_list, z_array_2D, 50, cmap='binary')
    #np.column_stack((alpha_list,beta_list))

    #surf:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Assign the data:
    X = np.array(alpha_list)
    Y = np.array(beta_list)
    X, Y = np.meshgrid(X, Y)
    Z = z_array_2D

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title('Fitness (Path Distance) vs. Alpha and Beta')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.show()


######## Test how the solution performs/converges with varying value for evap: ==========================================================
def evap_test(evap_test_list):
    list_distances = []
    plt.figure()
    fitness_vs_param_sweep("evap", evap_test_list,list_distances)
    ACO_results = plt.plot(evap_test_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(evap_test_list,[420] * len(evap_test_list), label="Optimal", linestyle='--', color='g')
    plt.title('Fitness (Path Distance) vs. Evap. (rho) Value')
    plt.xlabel('Evap Value')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()


###### Test how the solution performs/converges with varying evap2:===============================================================
def evap_test2(evap_test2_list):
    #evap_test2_list = [1, 10, 100, 500, 1000, 1500]
    list_distances =[]
    plt.figure()
    fitness_vs_param_sweep("evap2", evap_test2_list,list_distances)
    ACO_results = plt.plot(evap_test2_list,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    expected_results = plt.plot(evap_test2_list,[420] * len(evap_test2_list), label="Optimal", linestyle='--', color='g')
    plt.title('Fitness (Path Distance) vs. Evap Param 2')
    plt.xlabel('Evap -2')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()



##Multiple plots for comparison of different iterations/convergence =================================================================
def converge_vs_param_sweep(param, param_test, iter_num, list_distances, nodes = oliver30_nodes):
    for j in param_test:
        #Create colony with parameters of choice and solve for iter_num of iterations
        if(param == "alpha"):
            colony = ant_colony( nodes, distance, None, 50, j, 1.2, 0.4, 1000, iter_num)
        elif(param == "beta"):
            colony = ant_colony( nodes, distance, None, 50, 0.5, j, 0.4, 1000, iter_num)
        elif(param == "evap"):
            colony = ant_colony( nodes, distance, None, 50, 0.5, 1.2, j, 1000, iter_num)
        else:
            (print("not a valid type of parameter!"))
        answer = colony.mainloop()

        #Find Fitness (total distance from one node to the next in the path found):
        total_dist=0
        for i in range(len(answer)):
            if(i<len(answer)-1):
                node = answer[i]
                node2 = answer[i+1]
                total_dist+=colony.distance_matrix[node][node2]
            elif(i==len(answer)-1):
                node = answer[i]  #last
                node2 = answer[0] #first
                x=(colony.nodes[node][0])-(colony.nodes[node2][0])
                y=(colony.nodes[node][1])-(colony.nodes[node2][1])
                last_loop_term = abs(x) + abs(y)
                total_dist += last_loop_term

        ##Opt: print out the path and total distances found
        print(answer)
        print(total_dist)
        list_distances.append(total_dist)

    #Plot both our calculated ACO results against expected optimal performance for this benchmark
    #plt.plot(iter_test,list_distances,'-o')
    ACO_results = plt.plot(param_test,list_distances, label="Oliver30 ACO Results: " + str(iter_num), linestyle='-', marker='o')
    #expected_results = plt.plot(param_test,[420] * len(param_test), label="Optimal", linestyle='--', color='g')
    return plt


######Test multiple iterations with varying alpha... ---------------------------------------------------------------------

def alpha_iter_test(iter_test, alpha_test):
    iter_test = [1,4,10,50, 200]
    alpha_test = [0.1,0.25,0.5,0.75,1]
    plt.figure()
    for k in iter_test:
        list_distances =[]
        converge_vs_param_sweep("alpha", alpha_test, k, list_distances)
        plt.title('Fitness (Path Distance) vs. Alpha Value')
        plt.xlabel('Alpha Value')
        plt.ylabel('Fitness (Total distance)')
        plt.legend()
    expected_results = plt.plot(alpha_test,[420] * len(alpha_test), label="Optimal", linestyle='--', color='g')
    plt.show()

#####Test multiple iterations with varying evap/rho... --------------------------------------------------------------------

def iter_evap_test(iter_test, evap_test):
    #iter_test = [1,4,10,50,100]
    #evap_test = [0.1, 0.25, 0.4, 0.6, 0.8, 1]

    plt.figure()
    for k in iter_test:
        list_distances =[]
        converge_vs_param_sweep("evap", evap_test, k, list_distances)
        plt.title('Fitness (Path Distance) vs. Evap Value')
        plt.xlabel('Evap Value')
        plt.ylabel('Fitness (Total distance)')
        plt.legend()
    expected_results = plt.plot(evap_test,[420] * len(evap_test), label="Optimal", linestyle='--', color='g')
    plt.show()







### ==========================================================================================================================================
## 7. Ext 2: CSO Implementation Demonstration and Performance
### ==========================================================================================================================================

#Runs CSO on oliver30 node set, and outputs an iter vs. fitness/distance value (with optional step-through process node plots):
def CSO_run(iter_test, process_plot="False"):
    #iter_test = [1,5, 10, 50, 100, 300]
    dist_list = []
    for i in iter_test:
        colony = CSO(oliver30_nodes, 100, 10, 20, 5, i)
        answer = colony.cso_run()
        print(answer)
        calculated_dist = colony.eval(answer)
        print(calculated_dist)
        dist_list.append(calculated_dist)

        if(process_plot == True):
            print('test')
            # Plot Travelling Salesman Problem Nodes on X-Y Plot, and Distance:
            # https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp
            #plt.figure()

            #Make a 2D array of answer node x and y locations
            x_list=[]
            y_list = []
            #answer.append(answer[0])
            for ii in range(len(answer)):
                current_key = answer[ii]
                #print(current_key)
                #current_node = nodes[current_key]
                current_node = oliver30_nodes[current_key]
                #print(current_node)
                x_list.append(current_node[0])
                y_list.append(current_node[1])
            positions = np.column_stack((x_list, y_list))
            #print(positions)

            #Create 2 subplots - one for raw nodes, one for colored answer orders and edges
            fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
            ax[0].set_title('Raw nodes')
            ax[1].set_title('CSO Optimized Tour: Iter: ' + str(i))
            ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
            s=ax[1].scatter(positions[:, 0], positions[:, 1],c=np.arange(len(x_list)))             # plot B

            #colormap init
            YlGnBl = plt.get_cmap('YlGnBu')
            YlGnBlr= YlGnBl.reversed()
            RdPu = plt.get_cmap('RdPu')

            #for all the nodes in answer, plot
            start_node = 0
            distance1 = 0.
            N=len(answer)
            for k in range(N-1):
                start_pos = positions[start_node]
                #next_node = np.argmax(x_sol[start_node]) # needed because of MIP-approach used for TSP
                next_node = k+1
                #print(next_node)
                end_pos = positions[next_node]

                #Add arrows connecting edges of graph:
                # (switch scale based on num nodes - 8 (for Oliver30) to 25-50 for testnodes)
                ax[1].annotate("",
                        xy=start_pos, xycoords='data',
                        xytext=end_pos, textcoords='data',
                        arrowprops= dict(arrowstyle="->",
                                        connectionstyle="arc3",color=YlGnBlr(k*(8))))

                distance1 += np.linalg.norm(end_pos - start_pos)
                start_node = next_node

             # textbox overlay with distances:
            textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance1)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10, # Textbox
                    verticalalignment='top', bbox=props)
            plt.tight_layout()
            plt.show()


    CSO_results = plt.plot(iter_test,dist_list, label="Oliver30 CSO Results", linestyle='-', marker='o',color='r')
    expected_results = plt.plot(iter_test,[420] * len(iter_test), label="Optimal", linestyle='--', color='g')
    plt.title('Fitness (Path Distance) vs. Iterations')
    plt.xlabel('Num iterations')
    plt.ylabel('Fitness (Total distance)')
    plt.legend(loc="lower right")
    plt.show()



### ==========================================================================================================================================
## 8. Ext 3: CSO and ACO Comparisons
### ==========================================================================================================================================

#Function that sweeps a parameter with a given list of options and plots the performance of ACO:
def time_vs_param_sweep(param, param_test, list_distances, list_times, nodes=oliver30_nodes):
    for j in param_test:

        start = timeit.default_timer()

        #create based on parameter of interest:
        if(param == "iter"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, j)
        elif(param =="ant"):
            colony = ant_colony(nodes, distance, None, j, 0.5, 1.2, 0.4, 1000, 80)
        elif(param == "alpha"):
            colony = ant_colony(nodes, distance, None, 50, j, 1.2, 0.4, 1000, 80)
        elif(param == "beta"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, j, 0.4, 1000, 80)
        elif(param == "evap"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, j, 1000, 80)
        elif(param == "evap2"):
            colony = ant_colony(nodes, distance, None, 50, 0.5, 1.2, 0.4, j, 80)
        else:
            (print("not a valid type of parameter!"))

        answer = colony.mainloop()
        stop = timeit.default_timer()
        timer = stop-start
        print('Time: ', timer)

        #Find Fitness (total distance from one node to the next in the path found):
        total_dist = get_fitness(colony, answer)

        ##Optional Output: print out the path and total distances found
        print(answer)
        print(total_dist)
        list_distances.append(total_dist)

        #output time:
        list_times.append(timer)

    return plt


#### Test how the solution performs/converges with varying iterations when compared to ACO: -------------------------

def aco_cso_iter_fitness():
    ##iter_test = [1,2,4,10,50,200]
    iter_test = [1,2, 3, 4, 5, 10, 15, 25, 50, 100, 250]
    plt.figure()
    #iter_test = [1,2, 3, 4]
    list_distances =[]
    fitness_vs_param_sweep("iter", iter_test, list_distances, oliver30_nodes)
    ACO_results = plt.plot(iter_test,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    dist_list = []

    iter_test = [1,5, 10, 50, 100, 300]
    #iter_test = [1,2, 3, 4]
    CSO_run(iter_test,"False")

###### Test how the solution performs/converges with varying number of ants: ==========================================================

#ant_test = [1,2,4,10,50,100]

def aco_cso_ant_fitness():
    list_distances =[]

    fitness_vs_param_sweep("ant", ant_test, list_distances)
    plt.figure()
    ACO_results = plt.plot(ant_test,list_distances, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    #expected_results = plt.plot(ant_test,[420] * len(ant_test), label="Optimal", linestyle='--', color='g')
    dist_list = []
    for i in ant_test:
        colony = CSO(oliver30_nodes, i, 10, 20, 5, 80)
        answer = colony.cso_run()
        print(answer)
        calculated_dist = colony.eval(answer)
        print(calculated_dist)
        dist_list.append(calculated_dist)
    CSO_results = plt.plot(ant_test,dist_list, label="Oliver30 CSO Results", linestyle='-', marker='o',color='r')

    expected_results = plt.plot(ant_test,[420] * len(ant_test), label="Optimal", linestyle='--', color='g')
    plt.title('Fitness (Path Distance) vs. Number of Agents')
    plt.xlabel('Number Agents')
    plt.ylabel('Fitness (Total distance)')
    plt.legend()
    plt.show()


###### Test how the solution performs/converges with varying number of ants:TIME ==========================================================

#ant_test = [1,2,4,10,50,100]

def aco_cso_timing():
    list_distances =[]
    list_times =[]
    plt.figure()
    time_vs_param_sweep("ant", ant_test, list_distances, list_times)
    ACO_results = plt.plot(ant_test,list_times, label="Oliver30 ACO Results", linestyle='-', marker='o',color='b')
    #expected_results = plt.plot(ant_test,[420] * len(ant_test), label="Optimal", linestyle='--', color='g')
    dist_list = []
    list_times =[]
    for i in ant_test:
        start = timeit.default_timer()
        colony = CSO(oliver30_nodes, i, 10, 20, 5, 80)
        answer = colony.cso_run()
        print(answer)
        calculated_dist = colony.eval(answer)
        print(calculated_dist)
        dist_list.append(calculated_dist)
        stop = timeit.default_timer()
        timer = stop-start
        print('Time: ', timer)
        list_times.append(timer)
    CSO_results = plt.plot(ant_test,list_times, label="Oliver30 CSO Results", linestyle='-', marker='o',color='r')

    #expected_results = plt.plot(ant_test,[420] * len(ant_test), label="Optimal", linestyle='--', color='g')
    plt.title('Runtime vs. Number of Agents')
    plt.xlabel('Number Agents')
    plt.ylabel('Runtime')
    plt.legend()
    plt.show()



def CSO_run(iter_test, process_plot="False"):
    dist_list = []
    for i in iter_test:
        colony = CSO(oliver30_nodes, 100, 10, 20, 5, i)
        answer = colony.cso_run()
        print(answer)
        calculated_dist = colony.eval(answer)
        print(calculated_dist)
        dist_list.append(calculated_dist)

        if(process_plot == True):
            print('test')
            # Plot Travelling Salesman Problem Nodes on X-Y Plot, and Distance:
            # https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp
            #plt.figure()

            #Make a 2D array of answer node x and y locations
            x_list=[]
            y_list = []
            #answer.append(answer[0])
            for ii in range(len(answer)):
                current_key = answer[ii]
                print(current_key)
                #current_node = nodes[current_key]
                current_node = oliver30_nodes[current_key]
                print(current_node)
                x_list.append(current_node[0])
                y_list.append(current_node[1])
            positions = np.column_stack((x_list, y_list))
            print(positions)

            #Create 2 subplots - one for raw nodes, one for colored answer orders and edges
            fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
            ax[0].set_title('Raw nodes')
            ax[1].set_title('CSO Optimized Tour: Iter: ' + str(i))
            ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
            s=ax[1].scatter(positions[:, 0], positions[:, 1],c=np.arange(len(x_list)))             # plot B

            #colormap init
            YlGnBl = plt.get_cmap('YlGnBu')
            YlGnBlr= YlGnBl.reversed()
            RdPu = plt.get_cmap('RdPu')

            #for all the nodes in answer, plot
            start_node = 0
            distance1 = 0.
            N=len(answer)
            for k in range(N-1):
                start_pos = positions[start_node]
                #next_node = np.argmax(x_sol[start_node]) # needed because of MIP-approach used for TSP
                next_node = k+1
                #print(next_node)
                end_pos = positions[next_node]

                #Add arrows connecting edges of graph:
                # (switch scale based on num nodes - 8 (for Oliver30) to 25-50 for testnodes)
                ax[1].annotate("",
                        xy=start_pos, xycoords='data',
                        xytext=end_pos, textcoords='data',
                        arrowprops= dict(arrowstyle="->",
                                        connectionstyle="arc3",color=YlGnBlr(k*(8))))

                distance1 += np.linalg.norm(end_pos - start_pos)
                start_node = next_node

             # textbox overlay with distances:
            textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance1)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10, # Textbox
                    verticalalignment='top', bbox=props)
            plt.tight_layout()
            plt.show()


    CSO_results = plt.plot(iter_test,dist_list, label="Oliver30 CSO Results", linestyle='-', marker='o',color='r')
    expected_results = plt.plot(iter_test,[420] * len(iter_test), label="Optimal", linestyle='--', color='g')
    plt.title('Fitness (Path Distance) vs. Iterations')
    plt.xlabel('Num iterations')
    plt.ylabel('Fitness (Total distance)')
    plt.legend(loc="lower right")
    plt.show()






### ==========================================================================================================================================
## 9. ACO Method Baseline with added Spatial temporal Methods
### ==========================================================================================================================================

##ST_CHANGE_ADDED:
#test evap - a simple dict example, with just a single value (eg. could represent the total amount of time that location has spent in the sun)
#            (a more complicated ex could be a list with different values for different times - adjust the evap function to work accordingly).
#             right now manually put in random values for different points, but we could write a quick script to generate this for us
#             on the basis of node locations/arrangement (eg - populate the values based upon how a sun moves in the space).
test_evap = {0: 1, 1: 4, 2: 2, 3: 4, 4: 5, 5: 2, 6: 3, 7: 4, 8: 5, 9: 2}

#A function to get evaporation rate based on time and node...
def sun_travel(x, t, width = 5):
    if (abs(100 * np.sin(10* t)) + width) < x and x >= (abs(100 * np.sin(10 * t))- width):
        return 5
    else:
        return 0

#Returns the run time and nodes in sun based on speed, answer (l) and nodeset (n)
def time_in_sun(l, n, speed = 50):
    suntime = 0
    time = 0
    for i in range(len(l)-1):
        time += (distance(n[l[i]], n[l[i+1]]) / speed)
        if sun_travel(n[l[i]][1], time) != 0:
            suntime += 1
    return (time, suntime)



def st_vs_reg_comparison(process_plot = False):
    #ST: make a colony of ants with temporal info: ========================================================
    speed = 50
    colony = ant_colony_st(oliver30_nodes, distance, sun_travel, None, 50, 0.5, 1.2, speed, 0.4, 1000, 30)
    ##colony = ant_colony_st(test_nodes, distance, sun_travel, None, 50, 0.5, 1.2, 5, 0.4, 1000, 50)
    ##...that will find the optimal solution with ACO
    answer = colony.mainloop()
    print(answer)

    #Make a second regular colony (same as before - no ST effects) for comparison:
    colony2 = ant_colony(oliver30_nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, 30)
    #colony2 = ant_colony(oliver30_nodes, distance, sun_travel, None, 50, 0.5, 1.2, 5, 0.4, 1000, 30)
    ##colony = ant_colony_st(test_nodes, distance, sun_travel, None, 50, 0.5, 1.2, 5, 0.4, 1000, 50)
    ##...that will find the optimal solution with ACO
    answer2 = colony2.mainloop()
    print(answer2)

    print(time_in_sun(answer, oliver30_nodes, speed))
    print(time_in_sun(answer2, oliver30_nodes, speed))
    #================================================================================================================

    if(process_plot == True):
        # Plot Travelling Salesman Problem Nodes on X-Y Plot, and Distance:
        # https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp
        #plt.figure()
        x_list=[]
        y_list = []

        #Make a 2D array of answer node x and y locations
        #answer.append(answer[0])
        for ii in range(len(answer)):
            current_key = answer[ii]
            #print(current_key)
            #current_node = nodes[current_key]
            current_node = oliver30_nodes[current_key]
            #current_node = test_nodes[current_key]
            #print(current_node)
            x_list.append(current_node[0])
            y_list.append(current_node[1])
        positions = np.column_stack((x_list, y_list))
        #print(positions)

        #Make two subplots, one for raw nodes and one for the optimized tour
        fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
        ax[0].set_title('Raw nodes')
        ax[1].set_title('ASO-ST Optimized Tour')
        ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
        s=ax[1].scatter(positions[:, 0], positions[:, 1],c=np.arange(len(x_list))) # plot B

        #colormap:
        YlGnBl = plt.get_cmap('YlGnBu')
        YlGnBlr= YlGnBl.reversed()
        RdPu = plt.get_cmap('RdPu')

        #plot each point in N (number of nodes in the tour)
        start_node = 0
        distance1 = 0.
        N=len(answer)
        for k in range(N-1):
            start_pos = positions[start_node]
            #next_node = np.argmax(x_sol[start_node]) # needed because of MIP-approach used for TSP
            next_node = k+1
            #print(next_node)
            end_pos = positions[next_node]
            ax[1].annotate("",
                    xy=start_pos, xycoords='data',
                    xytext=end_pos, textcoords='data',
                    arrowprops= dict(arrowstyle="->",
                                    connectionstyle="arc3",color=YlGnBlr(k*(8)))) # switch to 25-50 for smaller sets
            distance1 += np.linalg.norm(end_pos - start_pos)
            start_node = next_node
        textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance1)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10, # Textbox
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()


    ## Test how the solution performs/converges with varying iterations - ST:===============================================================

def st_vs_reg_iter_comparison(process_plot = False):

        #ST: make a colony of ants with temporal info: ========================================================
        speed = 5
        colony = ant_colony_st(oliver30_nodes, distance, sun_travel, None, 50, 0.5, 1.2, speed, 0.4, 1000, 30)
        answer = colony.mainloop()

        #Make a second regular colony (same as before - no ST effects) for comparison:
        colony2 = ant_colony(oliver30_nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, 30)
        answer2 = colony2.mainloop()

        print(time_in_sun(answer, oliver30_nodes))
        print(time_in_sun(answer2, oliver30_nodes))

        #Init lists to store calcs:
        list_st_dist_avgs = []
        list_reg_dist_avgs = []
        list_st_suntimes_avgs = []
        list_reg_suntimes_avgs = []
        list_st_sunnodes_avgs = []
        list_reg_sunnodes_avgs = []

        #Set speeds and run params:
        #speed_list = [1, 5, 10]
        speed_list = [10, 20, 50]
        run_num = 20 #%take avg of this num of runs

        #for each speed in the list to sweep through:
        for j in speed_list:
            print('speed: ' + str(j))

            list_st_dist = []
            list_reg_dist = []
            list_st_suntimes = []
            list_reg_suntimes = []
            list_st_sunnodes = []
            list_reg_sunnodes = []

            for k in range((run_num)):
                print(k)
                speed = j
                colony = ant_colony_st(oliver30_nodes, distance, sun_travel, None, 50, 0.5, 1.2, speed, 0.4, 1000, 30)
                answer = colony.mainloop()
                colony2 = ant_colony(oliver30_nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, 30)
                answer2 = colony2.mainloop()
                sun_st = (time_in_sun(answer, oliver30_nodes, speed))
                sun_reg = (time_in_sun(answer2, oliver30_nodes, speed))
                fitness_st= get_fitness(colony, answer)
                fitness_reg=get_fitness(colony2, answer2)

                #results of fitness (distance), time in sun (calc. from dist. and speed), and nodes in sun:
                list_st_dist.append(fitness_st)
                list_reg_dist.append(fitness_reg)
                list_st_suntimes.append(sun_st[0])
                list_reg_suntimes.append(sun_reg[0])
                list_st_sunnodes.append(sun_st[1])
                list_reg_sunnodes.append(sun_reg[1])

            #averaged results of fitness/time/sun for a given speed:
            avg_st_dist = np.mean(list_st_dist)
            avg_reg_dist = np.mean(list_reg_dist)
            avg_st_suntimes = np.mean(list_st_suntimes)
            avg_reg_suntimes = np.mean(list_reg_suntimes)
            avg_st_sunnodes= np.mean(list_st_sunnodes)
            avg_reg_sunnodes = np.mean(list_reg_sunnodes)

            list_st_dist_avgs.append(avg_st_dist)
            list_reg_dist_avgs.append(avg_reg_dist)
            list_st_suntimes_avgs.append(avg_st_suntimes)
            list_reg_suntimes_avgs.append(avg_reg_suntimes)
            list_st_sunnodes_avgs.append(avg_st_sunnodes)
            list_reg_sunnodes_avgs.append(avg_reg_sunnodes)

            print("times run: " + str(run_num))
            print("ST dist: " + str(avg_st_dist))
            print("Regular dist: " + str(avg_reg_dist))
            print("ST suntimes: " + str(avg_st_suntimes))
            print("Regular suntimes: " + str(avg_reg_suntimes))
            print("ST sunnodes: " + str(avg_st_sunnodes))
            print("Regular sunnodes: " + str(avg_reg_sunnodes))

        #collection of fitness/time results of param sweep of speed:
        print("ST dist: " + str(list_st_dist_avgs))
        print("Regular dist: " + str(list_reg_dist_avgs))
        print("ST suntimes: " + str( list_st_suntimes_avgs))
        print("Regular suntimes: " + str(list_reg_suntimes_avgs))
        print("ST sunnodes: " + str(list_st_sunnodes_avgs))
        print("Regular sunnodes: " + str(list_reg_sunnodes_avgs))


        plt.figure
        plt.plot(speed_list, list_st_sunnodes_avgs, label="Spatial-temporal", color ='g')
        plt.plot(speed_list, list_reg_sunnodes_avgs, label="original", color ='b')
        plt.title('Nodes in Sun vs. Ant Speed (Avg. 20 runs)')
        plt.xlabel('Ant Speed')
        plt.ylabel('Nodes in Sun')
        plt.legend()
        plt.show()



###===================================================================================================================================
###Ext5: Investigate Effect of Uncertainty
# No obvious effects
#a function to get distance between nodes...
def distance_uncertain1(start, end, std_dev=0.1):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    x_distance_uncertain = np.random.normal(x_distance,std_dev)
    y_distance_uncertain = np.random.normal(y_distance,std_dev)
    return math.sqrt(pow(x_distance_uncertain, 2) + pow(y_distance_uncertain, 2))

#a function to get distance between nodes...
def distance_uncertain2(start, end, std_dev=1):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    x_distance_uncertain = np.random.normal(x_distance,std_dev)
    y_distance_uncertain = np.random.normal(y_distance,std_dev)
    return math.sqrt(pow(x_distance_uncertain, 2) + pow(y_distance_uncertain, 2))

#a function to get distance between nodes...
def distance_uncertain3(start, end, std_dev=5):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    x_distance_uncertain = np.random.normal(x_distance,std_dev)
    y_distance_uncertain = np.random.normal(y_distance,std_dev)
    return math.sqrt(pow(x_distance_uncertain, 2) + pow(y_distance_uncertain, 2))

#a function to get distance between nodes...
def distance_uncertain4(start, end, std_dev=15):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    x_distance_uncertain = np.random.normal(x_distance,std_dev)
    y_distance_uncertain = np.random.normal(y_distance,std_dev)
    return math.sqrt(pow(x_distance_uncertain, 2) + pow(y_distance_uncertain, 2))


def simple_example_ACO_uncertain():
    #given some nodes, and some locations...
    print(test_nodes)
    list_fitnesses = []
    list1_fitnesses = []
    list2_fitnesses = []
    list3_fitnesses = []
    list4_fitnesses = []

    for i in range(10):
        #...we can make a colony of ants...
        colony = ant_colony(oliver30_nodes, distance_uncertain1)
        #colony = ant_colony(test_nodes, distance, None, 50, 0.5, 1.2, 0.4, 1000, 10)
        #...to run ACO and get a tour
        answer = colony.mainloop()
        print(answer)
        total_dist = get_fitness(colony, answer)
        print(total_dist)
        list1_fitnesses.append(total_dist)

        colony = ant_colony(oliver30_nodes, distance_uncertain2)
        answer = colony.mainloop()
        print(answer)
        total_dist = get_fitness(colony, answer)
        print(total_dist)
        list2_fitnesses.append(total_dist)

        colony = ant_colony(oliver30_nodes, distance_uncertain3)
        answer = colony.mainloop()
        print(answer)
        total_dist = get_fitness(colony, answer)
        print(total_dist)
        list3_fitnesses.append(total_dist)

        colony = ant_colony(oliver30_nodes, distance_uncertain4)
        answer = colony.mainloop()
        print(answer)
        total_dist = get_fitness(colony, answer)
        print(total_dist)
        list4_fitnesses.append(total_dist)

        list_fitnesses = [np.mean(list1_fitnesses), np.mean(list2_fitnesses), np.mean(list3_fitnesses),np.mean(list4_fitnesses)]
        error = [np.std(list1_fitnesses), np.std(list2_fitnesses), np.std(list3_fitnesses),np.std(list4_fitnesses)]

        plt.figure
        plt.plot([0.1,1,5,10],list_fitnesses)
        plt.title('Fitness (Path Distance) vs. Uncertainty in Location')
        plt.xlabel('Uncertainty (sigma/stdev)')
        plt.ylabel('Fitness (Total distance)')
        plt.legend()
        plt.show()

        x_pos = np.arange(len(list_fitnesses))
        plt.figure
        fig, ax = plt.subplots()
        ax.bar(x_pos, list_fitnesses,
               yerr=error,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)
        plt.show()

    return list_fitnesses

###============================================================================================================================
## 10. MAIN - comment/uncomment sections to run below
###============================================================================================================================

##Initial Simple Testing/Baseline:
print('simple test: ')
simple_example_ACO()

print('\n Baseline - 3 Benchmarks: ')
iter_test_list = [1,2,4,10]
#iter_test = [1,2,3]
ant_test_list = [1,2,3]
three_benchmark_iter(ant_test_list)
three_benchmark_ant(iter_test_list)

#----------------------------------------------------------------

##EXT1: Parameter Sweeps:
print('\n Iteration test: ')
iter_test_list = [2, 5, 10]
#iter_test_list = [1,2,4,10,50,200]
iter_test(iter_test_list)

print('\n Ant test: ')
ant_test_list = [1,2,4,10,50,100]
ant_test(ant_test_list)

print('\n Alpha test: ')
alpha_test_list = [0.1,0.25,0.5,0.75,1]
alpha_test(alpha_test_list)

print('\n Beta test: ')
beta_test_list = [1, 1.2,2,3,4,5]
beta_test(beta_test_list)

print('\n Alpha-Beta test: ')
#alpha_list = [0.1,0.25,0.7]
#beta_list = [1, 2, 3]
alpha_list = [0.1, 0.15, 0.25, 0.4, 0.5, 0.6, 0.7, 0.85, 1]
beta_list = [1, 1.25, 1.5, 2, 3, 4, 5, 6, 7]
alpha_beta_test(alpha_list, beta_list)


print('\n Evap. test: ')
evap_test_list = [0.1, 0.25, 0.4, 0.6, 0.8, 1]
evap_test(evap_test_list)

print('\n Evap. test2: ')
evap_test2_list = [1, 10, 100, 500, 1000, 1500]
evap_test2(evap_test2_list)

print('\n Alpha iter test ')
iter_test = [1,4,10,50, 200]
alpha_test = [0.1,0.25,0.5,0.75,1]
alpha_iter_test(iter_test, alpha_test)

print('\n Beta iter test ')
iter_test = [1,4,10,50,100]
evap_test = [0.1, 0.25, 0.4, 0.6, 0.8, 1]
iter_evap_test(iter_test, evap_test)

#print('uncertain_test: ')
#colony_u=simple_example_ACO_uncertain()
#print(colony_u)


#----------------------------------------------------------------
###EXT 2 CSO:
iter_test = [1,5, 10, 50, 100, 300]
CSO_run(iter_test, True)

####EXT 3: CSO vs ACO: fitness iterations, fitness ants and timing
print('aco_cso_iter_fitness')
aco_cso_iter_fitness()

print('aco_cso_ant_fitness')
ant_test = [1,2,4,10,50,100]
aco_cso_ant_fitness()

print('aco_cso_ant_timing')
ant_test = [1,2,4,10,50,100]
aco_cso_timing()

#----------------------------------------------------------------
###Ext 4: Spatial Temporal.
print('st_vs_reg_comparison')
st_vs_reg_comparison() ##include True to have node map show

print('st_vs_reg_iter_comparison:')
st_vs_reg_iter_comparison()
