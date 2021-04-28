from __future__ import division
from random import shuffle
import sys
from non_dominated_sort import *
import math
import time
from utility import *
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


lessismore = {}
lessismore["results_A/"] = [True, True]
lessismore["results_B/"] = [True, True]
lessismore["results_C/"] = [False, False]
lessismore["results_D/"] = [True, True]
lessismore["results_E/"] = [True, True]
lessismore["results_F/"] = [False, True]
lessismore["results_G/"] = [True, True]
lessismore["results_H/"] = [True, True]
lessismore["results_I/"] = [True, True]
lessismore["results_J/"] = [True, True]
lessismore["results_K/"] = [True, True]

ranges = {}
ranges["A"] = [[0,1],[0,1]]
ranges["B"] = [[0,1],[0,1]]
ranges["C"] = [[0,1],[0,1]]
ranges["D"] = [[0,1],[0,1]]
ranges["E"] = [[0,1],[0,1]]
ranges["F"] = [[0,1],[0,1]]
ranges["G"] = [[0,1],[0,1]]
ranges["H"] = [[0,1],[0,1]]
ranges["I"] = [[0,1],[0,1]]
ranges["J"] = [[0,1],[0,1]]
ranges["K"] = [[0,1],[0,1]]

def same_list(list1, list2):
    assert(len(list1) == len(list2)), "Something is wrong"
    for i, j in zip(list1, list2):
        if i!=j: return False
    return True


def not_in_cache(list, listoflist):
    for l in listoflist:
        if same_list(list, l) is True:
            return False
    return True

def normalize(x, mins, maxs):
    tmp = float((x - mins)) / (maxs - mins + 0.000001)
    if tmp > 1: return 1
    elif tmp < 0: return 0
    else: return tmp


def loss(x1, x2, mins=None, maxs=None):
    # normalize if mins and maxs are given
    if mins and maxs:
        x1 = [normalize(x, mins[i], maxs[i]) for i, x in enumerate(x1)]
        x2 = [normalize(x, mins[i], maxs[i]) for i, x in enumerate(x2)]

    o = min(len(x1), len(x2))  # len of x1 and x2 should be equal
    return sum([-1*math.exp((x1i - x2i) / o) for x1i, x2i in zip(x1, x2)]) / o


def get_cdom_values(objectives, lessismore):
    dependents = []
    for rd in objectives:
        temp = []
        for i in range(len(lessismore)):
            # if lessismore[i] is true - Minimization else Maximization
            if lessismore[i] is False:
                temp.append(1/rd[i])
            else:
                temp.append(rd[i])
        dependents.append(temp)

    maxs = []
    mins = []
    for i in range(len(objectives[0])):
        maxs.append(max([o[i] for o in dependents]))
        mins.append(min([o[i] for o in dependents]))

    cdom_scores = []
    for i, oi in enumerate(dependents):
        sum_store = 0
        for j, oj in enumerate(dependents):
            if i!=j:
                if loss(oi, oj, mins, maxs) < loss(oj, oi, mins, maxs):
                    sum_store += 1
        cdom_scores.append(sum_store)
    return cdom_scores


def get_nd_solutions(file, train_indep, training_dep, testing_indep,min_split=10,impurity_decrease=0.01):
    no_of_objectives = len(training_dep[0])
    predicted_objectives = []
    for objective_no in range(no_of_objectives):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(criterion='mse',
                                      min_samples_split=min_split,min_impurity_decrease=impurity_decrease)
        model.fit(train_indep, [t[objective_no] for t in training_dep])
        predicted = model.predict(testing_indep)
        predicted_objectives.append(predicted)

    # Merge the objectives
    merged_predicted_objectves = []
    for i, j in zip(predicted_objectives[0], predicted_objectives[1]):
        merged_predicted_objectves.append([i, j])
    assert(len(merged_predicted_objectves) == len(predicted_objectives[0])), "Something is wrong"

    # Find Non-Dominated Solutions
    pf_indexes = non_dominated_sort(merged_predicted_objectves, lessismore['results_' + file + '/'],[r[0] for r in ranges[file]],
                                             [r[1] for r in ranges[file]])
    # print "Number of ND Solutions: ", len(pf_indexes)

    return [testing_indep[i] for i in pf_indexes], [merged_predicted_objectves[i] for i in pf_indexes]


def get_training_sequence(file, training_indep, training_dep, testing_indep, after=None,min_split=10,impurity_decrease=0.01):
    # build a model and get the predicted non dominated solutions
    return_nd_independent, predicted_objectives = get_nd_solutions(file, training_indep, training_dep,
                                                  testing_indep,min_split,impurity_decrease=impurity_decrease)
    cdom_scores = get_cdom_values(predicted_objectives, lessismore['results_' + file + '/'])
    if not after:
        training_sequence = [i[0] for i in sorted(enumerate(cdom_scores), key=lambda x:x[1], reverse=False)]
    else:
        combined = []
        for i in range(len(predicted_objectives)):
            combined.append([cdom_scores[i], np.max(euclidean_distances(after,[predicted_objectives[i]]))])
        training_sequence = np.argsort([each[0] for each in sorted(combined, key = lambda x : x[1], reverse=False)])
    return training_sequence, return_nd_independent


def get_training_sequence_naive(file, training_indep, training_dep, testing_indep,min_split=10,impurity_decrease=0.01):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(criterion='mse',
                                  min_samples_split=min_split, min_impurity_decrease=impurity_decrease)
    model.fit(training_indep, training_dep)
    y_predicted = model.predict(testing_indep)
    pf_indexes = non_dominated_sort(y_predicted, lessismore['results_' + file + '/'],
                                    [r[0] for r in ranges[file]],
                                    [r[1] for r in ranges[file]])
    return [testing_indep[i] for i in pf_indexes], [y_predicted[i] for i in pf_indexes]



def hypervolume(raw_dependents, lessismore, mins, maxs):
    l = [0 for i in range(len(raw_dependents))]
    for i in range(len(raw_dependents)):
        if lessismore[i]:
            l[i] = maxs[i]+1 - raw_dependents[i]
        else:
            l[i] = raw_dependents[i] - (mins[i]-1)
    return np.prod(l)


def get_nd_layers(merged_pred,lessismore,hv=False):
    if hv:
        return [hypervolume(each,lessismore,[r[0] for r in ranges[file]],
                           [r[1] for r in ranges[file]]) for each in merged_pred]
    ideal=[]
    for i in range(len(lessismore)):
        if lessismore[i]:
            ideal.append(0)
        else:
            ideal.append(1)
    return euclidean_distances([ideal],merged_pred)


def get_cdom_layers(merged_pred,lessismore):
    vals = get_cdom_values(merged_pred,lessismore)
    return np.argsort(vals)


def abcd(pred_ys,actual_ys,lessismore):
    solution_index = []
    for k in range(len(lessismore)):
        if lessismore[k]:
            m = min(pred_ys[k])
            solution_index.append([i for i, j in enumerate(pred_ys[k]) if j == m])
        else:
            m = max(pred_ys[k])
            solution_index.append([i for i, j in enumerate(pred_ys[k]) if j == m])
    # print("FIXME",len(pred_ys[0]),len(pred_ys[1]))
    a = np.mean([actual_ys[0][i] for i in solution_index[0]])
    b = np.mean([actual_ys[1][i] for i in solution_index[0]])
    c=  np.mean([actual_ys[0][i] for i in solution_index[1]])
    d = np.mean([actual_ys[1][i] for i in solution_index[1]])

    a_sd = np.std([actual_ys[0][i] for i in solution_index[0]])
    b_sd = np.std([actual_ys[1][i] for i in solution_index[0]])
    c_sd = np.std([actual_ys[0][i] for i in solution_index[1]])
    d_sd = np.std([actual_ys[1][i] for i in solution_index[1]])

    return [a,b,c,d],[a_sd,b_sd,c_sd,d_sd]


def get_ranks(pred_ys,lessismore):
    ranks=[]
    for k in range(len(lessismore)):
        if lessismore[k]:
            a = pd.DataFrame(pred_ys[k])
            ranks.append(a.rank(ascending=True).iloc[:, 0].values)
        else:
            a = pd.DataFrame(pred_ys[k])
            ranks.append(a.rank(ascending=False).iloc[:, 0].values)
    return ranks

def mrd(rank1,rank2):
    delta = 0
    for i in range(len(rank1)):
        delta+=abs(rank1[i]-rank2[i])
    return np.mean(delta)/len(rank1)



if __name__ == "__main__":
    from utility import read_file, split_data, build_model
    # files =["C","E","F","G","J","SS-J","SS-L"]
    # files = [ "J","SS-J","SS-L"]
    # files = ["G"]
    files=["D","K"]
    for file in files:
        print(file)
        df = pd.read_csv('../Data/' + file + '.csv')
        y = df.iloc[:, -2:]
        X = df.iloc[:, :-2]

        print("RANGE:",df.iloc[:,-2].min(),df.iloc[:,-2].max(),df.iloc[:,-1].min(),df.iloc[:,-1].max())
        column_names = list(df.columns[:-2])
        mm = MinMaxScaler()
        y = mm.fit_transform(y)
        X = mm.fit_transform(X)

        N = df.shape[1] - 2

        size_PF=[]
        c1g2=[]
        c1g2_std=[]
        c2g1=[]
        c2g1_std=[]
        mean_selected1s=[]
        mean_selected2s=[]

        ranks1=[]
        ranks2=[]

        # performance_drop1,performance_drop2=[],[]
        dum1,dum2,dum11,dum22=[],[],[],[]
        time1,time2= [],[]
        train1,train2=[],[]
        win,tie,lose = 0,0,0
        objectives_dict = {}

        data = read_file1('../Data/' + file + '.csv')
        for d in data:
            key = ",".join(map(str, d.decisions))
            objectives_dict[key] = d.objectives


        def get_objective_score(independent):
            key = ",".join(map(str, independent))
            return objectives_dict[key]

        M=3100
        split=2
        impurity=0.00001
        for rep in range(M):
            start = time.time()
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5,random_state=rep)
            X_valid = pd.DataFrame(X_valid)
            training_dep = pd.DataFrame(y_train).values[:50]
            training_indep = pd.DataFrame(X_train).values[:50]
            testing_dep = pd.DataFrame(y_train).values[50:]
            testing_indep = pd.DataFrame(X_train).values[50:]
            lives = 20
            model = DecisionTreeRegressor(criterion='mse', min_samples_split=split, min_impurity_decrease=impurity)
            while True:
                training_dep = [get_objective_score(r) for r in training_indep]
                testing_dep = [get_objective_score(r) for r in testing_indep]
                training_sequence, return_nd_independent = get_training_sequence(file, training_indep, training_dep,
                                                            testing_indep,min_split=split,impurity_decrease=impurity)
                next_point = return_nd_independent[training_sequence[0]]
                next_point_dependent = get_objective_score(next_point)
                before_pf_indexes = non_dominated_sort(training_dep, lessismore['results_' + file + '/'],
                                                       [r[0] for r in ranges[file]], [r[1] for r in ranges[file]])
                before_pf = [training_dep[i] for i in before_pf_indexes]

                added_training = training_indep + [next_point]
                after_pf_indexes = non_dominated_sort(training_dep + [next_point_dependent],
                                                      lessismore['results_' + file + '/'], [r[0] for r in ranges[file]],
                                                      [r[1] for r in ranges[file]])
                after_pf = [(list(training_dep) + list([next_point_dependent]))[i] for i in after_pf_indexes]
                after_pf_indep = [(list(training_dep) + list([next_point_dependent]))[i] for i in after_pf_indexes]
                import itertools
                after_pf = [list(each) for each in after_pf]
                after_pf.sort()
                after_pf = [k for k, _ in itertools.groupby(after_pf)]
                after_pf_indep = [list(each) for each in after_pf_indep]
                after_pf_indep.sort()
                after_pf_indep = [k for k, _ in itertools.groupby(after_pf_indep)]
                # See if the new point is a dominant point
                previously_seen = []
                previously_not_seen = []
                for cr in after_pf:
                    seen = False
                    for pr in before_pf:
                        # Previously Seen
                        if same_list(pr, cr):
                            seen = True
                            previously_seen.append(cr)
                            continue
                    if seen is False:
                        previously_not_seen.append(cr)

                if len(previously_not_seen) == 0:
                    lives -= 1
                # print("lives", lives,len(training_indep))
                # print("Size of the frontier = ", len(after_pf))
                training_indep = list(training_indep) + list([next_point])
                for i in range(len(testing_indep)):
                    if same_list(testing_indep[i], next_point):
                        testing_indep = np.delete(testing_indep, i, 0)
                        break
                if lives == 0 or len(testing_indep) == 0: break
            training_dep = [get_objective_score(r) for r in training_indep]
            testing_dep = [get_objective_score(r) for r in testing_indep]
            y_test = testing_dep.copy()
            X_test = pd.DataFrame(testing_indep).copy()
            y_train = training_dep.copy()
            X_train = pd.DataFrame(training_indep).copy()
            duration = time.time()-start
            train1.append(duration)
            # print("Training time 1:",duration)
            print("Refined space ratio:",len(y_train),len(y_test))

            y_train1 = pd.DataFrame(y_train).iloc[:, 0]
            y_train2 = pd.DataFrame(y_train).iloc[:, 1]

            gd1=[]
            gd2=[]
            start = time.time()
            X_train_reduced = X_train.iloc[:, :]
            X_test_reduced = X_valid.iloc[:, :]
            model = DecisionTreeRegressor(criterion='mse',min_samples_split=split,min_impurity_decrease=impurity)

            model.fit(pd.DataFrame(X_train_reduced), pd.DataFrame(y_train).iloc[:, 0])
            y1 = model.predict(pd.DataFrame(X_train_reduced))
            y_pred1 = model.predict(pd.DataFrame(X_test_reduced))

            model.fit(pd.DataFrame(X_train_reduced), pd.DataFrame(y_train).iloc[:, 1])
            y2 =  model.predict(pd.DataFrame(X_train_reduced))
            y_pred2 = model.predict(pd.DataFrame(X_test_reduced))

            merged = pd.DataFrame([y_pred1, y_pred2]).T
            pf_indexes = non_dominated_sort(merged.values, lessismore['results_' + file + '/'],
                                            [r[0] for r in ranges[file]],
                                            [r[1] for r in ranges[file]])
            current_pf = [y_valid[i] for i in pf_indexes]
            duration = time.time() - start

            pred_ys = [y1,y2]
            actual_ys = [pd.DataFrame(y_train).iloc[:, 0], pd.DataFrame(y_train).iloc[:, 1]]
            rq1_mean, rq1_std = abcd(pred_ys,actual_ys, lessismore['results_' + file + '/'])
            print("Confusion matrix 1:",rq1_mean)

            # mean_selected1s.append(rq1_mean[0])
            # mean_selected2s.append(rq1_mean[3])
            # c1g2.append(rq1_mean[1])
            # c2g1.append(rq1_mean[2])
            # c1g2_std.append(rq1_std[1])
            # c2g1_std.append(rq1_std[2])
            r = get_ranks(pred_ys, lessismore['results_' + file + '/'])
            ranks1.append(r[0])
            ranks2.append(r[1])
            g1 = np.mean([each[0] for each in current_pf])
            g2 = np.mean([each[1] for each in current_pf])
            print("***FLASH results in 2 goals:", g1, g2)

            time1.append(duration)
            print("Time1:", duration,"Size:",len(current_pf))

            pf_indexes = non_dominated_sort(y_valid, lessismore['results_' + file + '/'], [r[0] for r in ranges[file]],
                                            [r[1] for r in ranges[file]])
            true_pf = [y_valid[i] for i in pf_indexes]
            gd = (generational_distance(true_pf, current_pf, ranges[file]))
            dum1.append(gd)
            igd = (inverted_generational_distance(true_pf, current_pf, ranges[file]))
            dum11.append(igd)
            size = len(current_pf)

            start=time.time()
            if X_test.shape[0]<=2**N:
                print("Undersample")
                X_fake = pd.DataFrame(X_test)
            else:
                X_fake = X_test.sample(n=2**N,replace=False)

            y_predicted = []
            for i in range(2):
                model = DecisionTreeRegressor(criterion='mse',min_samples_split=split,min_impurity_decrease=impurity)
                model.fit(X_train, pd.DataFrame(y_train).iloc[:, i])
                y_predicted.append(model.predict(X_fake))
            merged_pred = []
            for i, j in zip(y_predicted[0], y_predicted[1]):
                merged_pred.append([i, j])
            layers = get_nd_layers(merged_pred, lessismore=lessismore['results_' + file + '/'])
            labels = layers[0]
            labels= rankdata(labels)
            duration=time.time()-start
            train2.append(duration)
            start = time.time()
            outter_model = DecisionTreeRegressor(criterion='mse',min_samples_split=split,
                                                 min_impurity_decrease=impurity)
            outter_model.fit(X_fake, labels)

            y_pred = outter_model.predict(X_test_reduced)
            winner = np.argwhere(y_pred == np.amin(y_pred))
            pf_indexes = winner.flatten().tolist()
            current_pf = [y_valid[i] for i in pf_indexes]
            g1 = np.mean([each[0] for each in current_pf])
            g2 = np.mean([each[1] for each in current_pf])
            print("***VEER results in 2 goals:", g1,g2 )
            duration = time.time() - start
            time2.append(duration)

            print("Time2:", duration,"Size:",len(current_pf),len(true_pf))
            gd = (generational_distance(true_pf, current_pf, ranges[file]))
            dum2.append(gd)
            igd = (inverted_generational_distance(true_pf, current_pf, ranges[file]))
            dum22.append(igd)
            size_PF.append(len(current_pf))
            print("GD for FLASH vs VEER:",dum1[-1],dum2[-1])
            print("IGD for FLASH vs VEER:", dum11[-1], dum22[-1])
            print(file,rep+1)
            print()
        print("="*60,"\n")
        tag = "_sample"
        out = pd.DataFrame(dum1).T
        out.to_csv(file + tag + "_gd_flash.csv", sep=',')
        out = pd.DataFrame(dum2).T
        out.to_csv(file + tag + "_gd_veer.csv", sep=',')

        out = pd.DataFrame(time1).T
        out.to_csv(file + tag + "_testing_time_flash.csv", sep=',')
        out = pd.DataFrame(time2).T
        out.to_csv(file + tag + "_testing_time_veer.csv", sep=',')

        out = pd.DataFrame(train1).T
        out.to_csv(file + tag + "_training_time_flash.csv", sep=',')
        out = pd.DataFrame(train2).T
        out.to_csv(file + tag + "_training_time_veer.csv", sep=',')

        out = pd.DataFrame(ranks1).T
        out.to_csv(file + tag + "_rank1.csv", sep=',')
        out = pd.DataFrame(ranks2).T
        out.to_csv(file + tag + "_rank2.csv", sep=',')
