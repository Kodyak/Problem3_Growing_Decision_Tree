import math
# assumes pandas dataset and numpy methods

### basic functions

# reimplementing basic dataset subseting to use in building decision tree
def subset(dataset, attribute, value):
    """
    Takes in a dataset, and attribute (column) and value, 
    returns the rows from dataset that cofrrespond to where attribute = value as a sub-dataset
    """
    sub_dataset = dataset.loc[dataset[attribute] == value]
    return sub_dataset

### information gain

# info entropy
def p_logp(p): 
    """
    scope --- p is proportion from 0 to 1
    """
    if (p < 10**(-10)) | (p > 1 - 10**(-10)) :
        return 0
    else: return (-1) * p * math.log2(p)

    
# information measure
def info_measure(v): 
    """
    NOTE: vector should be of counts: all terms in it ≥ 0
    """    
#     if v == [0]*len(v):  return 0
    if sum(v) == 0:  return 0
    return sum( [ p_logp( s/sum(v) ) for s in v ] )

def compute_info_purity(v):
    """
    used for computing info purity for a target vector 
    (combines class values together then uses info measure)
    """
    ### get label counts
    label_counts = []
    for label_value in set(v):
        # pick out rows from dataset with desired label value
        sub_data = [x for x in v if x==label_value]
        counts = len(sub_data)
        label_counts.append(counts)  #   
        
    return info_measure(label_counts)

# entropy cost associated with an attribute
# ASSUME: last column of input dataset is 'count' associated with particular outcome
# ASSUME: first column is class label
def entropy(dataset, attribute, debug_flag=False):
    label = dataset.iloc[:,-1] # 1st column class label
    total_counts = dataset.shape[0]  # number of rows
    total_entropy = 0
    
    for value in set(dataset[attribute]):
        if debug_flag: print(attribute, value) #debugging
        sub_data = dataset.loc[dataset[attribute] == value]
        if debug_flag: print(sub_data) #debugging
        
        label_counts = []
        for label_value in set(label):
            # pick out rows from sub data with desired label value
            sub_sub_data = sub_data.loc[sub_data.iloc[:,-1] == label_value] 
            label_counts.append(sub_sub_data.shape[0])  # REPLACE 'count' eventually

#         if debug_flag: print(label_counts) #debugging
        if debug_flag: print(str(sum(label_counts)) + '/' + str(total_counts) + ' * I(' + str(label_counts) + ')') #debugging


        total_entropy += (sum(label_counts)/total_counts) * info_measure(label_counts)
    if debug_flag: print('*'*50)
    return total_entropy


# Define Information Gain Criteria 
# equivalent to minimizing entropy but has a nice interpretation
def info_gain(dataset, attribute, debug_flag = False):
    
    #get label counts
    label = dataset.iloc[:,-1] # assume it's last column for now
    label_counts = []
    
    for label_value in set(label):
        # pick out rows from dataset with desired label value
        sub_data = dataset.loc[dataset.iloc[:,-1] == label_value] 
        label_counts.append(sub_data.shape[0])  # 

    if debug_flag: 
        print('I(...):', label_counts) #debugging
        print('*'*50) #debugging
    
    I = info_measure(label_counts)
    return I - entropy(dataset, attribute, debug_flag)