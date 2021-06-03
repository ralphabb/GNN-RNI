import networkx as nx
import os
import subprocess
from networkx.algorithms import bipartite
import PyMiniSolvers.minisolvers as minisolvers
from tqdm import tqdm
import numpy as np
import copy
import torch
from torch_geometric.data import Data
import pickle
import matplotlib
from typing import List
matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
PLANTRI_DIR = "plantri/plantri"
TEMPORARY_FILE = "temp.out"
MAX_WIDTH_THRESHOLD = 5
MIN_WIDTH_THRESHOLD = 2
DATASET_NAME = "GRAPHSAT"


# Visualisation function to draw the chain in the dataset core pair
def draw_chain(starting_v_idx, starting_c_idx, chain_length, graph=None, return_pos=False, direction=False):
    if not direction:  # Left to right chain orientation (Set positions for literals and their negations (V-))
        pos = {"V"+str(starting_v_idx + nb): (0.07 + nb*(0.84/(chain_length-1)), 0.8) for nb in range(chain_length)}
        pos.update({"V-"+str(starting_v_idx + nb): (0.13 + nb*(0.84/(chain_length-1)), 0.9)
                    for nb in range(chain_length)})
    else:  # Right to left chain orientation
        pos = {"V" + str(starting_v_idx + nb): (0.13 + nb * (0.84 / (chain_length - 1)), 0.8) for nb in
               range(chain_length)}
        pos.update({"V-" + str(starting_v_idx + nb): (0.07 + nb * (0.84 / (chain_length - 1)), 0.9)
                    for nb in range(chain_length)})
    # Conjunction drawing
    pos.update({"C"+str(starting_c_idx + nb): (0.2 + nb*(0.6/(chain_length-1)), 0.1) for nb in range(chain_length)})
    if not return_pos and graph is not None:
        nx.draw(graph, pos)
        plt.show()
    else:
        return pos


def affine_transform_positions(pos_dict, x_scale=0.5, y_scale=0.5, x_bias=0.0, y_bias=0.0):  # Helper Functions
    return {x: (x_scale * y[0] + x_bias, y_scale * y[1] + y_bias) for x, y in pos_dict.items()}


# Returns the conjuncts for the 'bridge' structure in the core pair
def add_overarching_conj(core_graph: nx.Graph, starting_v_idx, starting_c_idx, chain_length, return_pos=True):
    hc_len = chain_length // 2  # Half-Chain Length
    # Chain length has to be an even number
    new_conj_names = ["C"+str(starting_c_idx+chain_length+nb) for nb in range(chain_length)]   # New Conjunct Names
    core_graph.add_nodes_from(new_conj_names, label="c")
    new_edges = [("V"+str(starting_v_idx+nb), new_conj_names[get_conj_index(nb, hc_len)])
                 for nb in range(chain_length)]  # +ve nodes, even conj idx.
    new_edges.extend([("V-"+str(starting_v_idx+nb), new_conj_names[1 + get_conj_index(nb, hc_len)])
                      for nb in range(chain_length)])
    core_graph.add_edges_from(new_edges)
    if return_pos:  # Give positions for drawings
        pos = {disjunction: (0.5, 0.9 - 0.4*idx/(chain_length - 1)) for idx, disjunction in enumerate(new_conj_names)}
        return core_graph, pos
    return core_graph, None  # Returns an updated core graph, and None if no position required


# Main core generation function
def generate_cores(strt_v_idx, strt_c_idx, chain_length, draw=True):
    assert chain_length % 2 == 0    # Ensure that the provided chain length is even
    # Part I: The SAT Core
    nb_divisions = 2  # The chain is divided into two parts
    sat_core = nx.Graph()  # SAT Core first (2 symmetric chains)
    chain_size = chain_length // 2  # The two parts of the chain are equally long
    sat_chain_pos, unsat_chain_pos = {}, {}  # For drawing
    for division in range(nb_divisions):   # Now produce the divisions
        chain_direction = bool(division % 2)  # False if 0, True if 1 -> one chain in the positive dir, other negative
        subchain = generate_chain(strt_v_idx + division * chain_size,
                                  strt_c_idx + division * chain_size, chain_size, direction=chain_direction)
        sat_core.add_nodes_from(subchain.nodes(data=True))
        sat_core.add_edges_from(subchain.edges())
        if draw:    # Additional drawing position passing (affine transform to fit both drawings of chain divisions)
            sat_chain_pos.update(affine_transform_positions(draw_chain(strt_v_idx + division * chain_size,
                                                                       strt_c_idx + division * chain_size, chain_size,
                                                                       return_pos=True, direction=chain_direction),
                                                            x_bias=(1/nb_divisions + 0.05)*division), x_scale=0.45)
    # Now add the bridge conjunctions
    sat_core, sat_pos = add_overarching_conj(sat_core, strt_v_idx, strt_c_idx, chain_length, return_pos=draw)
    if draw:
        sat_chain_pos.update(sat_pos)
        nx.draw(sat_core, sat_chain_pos)
        plt.show()
    # Part II: UNSAT Core --- Here it's just one single chain
    unsat_core = generate_chain(strt_v_idx, strt_c_idx, chain_length, direction=False)  # Full chain
    if draw:
        unsat_chain_pos = affine_transform_positions(draw_chain(strt_v_idx, strt_c_idx, chain_length,
                                                                return_pos=True, direction=False),
                                                     x_scale=1.0, y_scale=0.5, x_bias=0, y_bias=0)
    # Add bridge
    unsat_core, unsat_pos = add_overarching_conj(unsat_core, strt_v_idx, strt_c_idx, chain_length, return_pos=draw)
    if draw:
        unsat_chain_pos.update(unsat_pos)
        nx.draw(unsat_core, unsat_chain_pos)
        plt.show()
        # print(sat_check_from_graph(6, 12, sat_core))
    return sat_core, unsat_core


def get_conj_index(nb, hc_len):  # Helper function to retrieve conjunctions, chain_length is an even number
    # hc_len: half-chain length
    # nb is between 0 (inclusive) ad 2 * hc_len (exclusive)
    integer_div = nb // hc_len  # Either 0 or 1 (which division is it part of)
    multiplier = 1 - 2 * integer_div   # If nb >= hc_len, -1 else 1
    return 2*((multiplier * nb - integer_div) % hc_len)


# Produce the chain part that underpins both instances of the core pair
def generate_chain(starting_v_idx, starting_c_idx, chain_length, direction=False):  # Chain constructor
    chain = nx.Graph()  # Graph Instantiation
    # Node Addition (Positive and negative literals, then conjuncts)
    chain.add_nodes_from(["V"+str(nb) for nb in range(starting_v_idx, starting_v_idx + chain_length)] +
                         ["V-"+str(nb) for nb in range(starting_v_idx, starting_v_idx + chain_length)], label="v")
    chain.add_nodes_from(["C"+str(nb) for nb in range(starting_c_idx, starting_c_idx + chain_length)], label="c")
    # Edge Addition
    chain.add_edges_from([("V-" + str(starting_v_idx + idx),
                           "C" + str(starting_c_idx + idx)) for idx in range(chain_length)])
    if not direction:  # Standard left to right:
        chain.add_edges_from([("V" + str(starting_v_idx + (idx + 1) % chain_length),
                               "C" + str(starting_c_idx + idx)) for idx in range(chain_length)])
    else:  # Right to left direction
        chain.add_edges_from([("V" + str(starting_v_idx + (idx - 1) % chain_length),
                               "C" + str(starting_c_idx + idx)) for idx in range(chain_length)])
    # Literal-Literal Connection
    chain.add_edges_from([("V"+str(nb), "V-"+str(nb)) for nb in range(starting_v_idx, starting_v_idx + chain_length)])
    # draw_chain(chain, starting_v_idx, starting_c_idx, chain_length, direction)
    return chain


def list_helper(input_list: List):  # Helper function
    if len(input_list) > 1:
        return input_list[1:]
    else:
        return []   # Prevent exception


# Verify the satisfiability / unsatisfiability of generated instances
def sat_check_from_graph(nb_var, nb_conj, graph: nx.Graph, verbose=False):
    # May be suboptimal, but since nb variables are different, and it's not clear how to "reset" the solver
    sat_solver = minisolvers.MinisatSolver()   # Load the MiniSat solver
    for i in range(nb_var):
        sat_solver.new_var(dvar=True)  # Instantiate the necessary variables
    iclauses = [[] for _ in range(nb_conj)]
    for edge in nx.edges(graph):  # Build the MiniSAT format formula to be fed into the solver
        if edge[0][0] == "C":  # Intra-Variable edges haven't been introduced yet, so this is guaranteed
            idx_to_append = int(edge[0][1:]) - 1  # Parse the index of the conjunct from its name
            iclauses[idx_to_append].append(int(edge[1][1:]))    # Now get the corresponding variable
        elif edge[1][0] == "C":
            idx_to_append = int(edge[1][1:]) - 1
            iclauses[idx_to_append].append(int(edge[0][1:]))
    for iclause in iclauses:
        if len(iclause) > 0:  # Eliminate redundant (empty) clauses corresponding to removed disjunctions
            sat_solver.add_clause(iclause)  # Add the clauses
    if verbose:
        print(iclauses)
    result = sat_solver.solve()  # Solve the SAT instance
    return result


def process_graph(graph: nx.Graph, embedding: dict):  # Processing a given Plantri graph G
    if nx.is_bipartite(graph):  # Sanity Check: Is the graph bipartite? (Unnecessary, but mostly to get the BP sets)
        set1, set2 = bipartite.sets(graph)  # Step 1: Get the bipartite sets
        disjunctions, variables = (set1, set2) if len(set1) < len(set2) else (set2, set1)   # Conjuncts --> smaller set
        # Note: Every conjunct is a disjunction, and we refer to conjuncts as disjunctions henceforth
        nb_var, nb_disj = len(variables), len(disjunctions)  # Number of ``variables'' in the graph
        disjunction_widths = [y[1] for y in graph.degree(disjunctions)]
        max_disj_width = max(disjunction_widths)
        if max_disj_width > MAX_WIDTH_THRESHOLD:  # Step 2: Check relative to maximum width (min w handled by plantri)
            # 2.1: Find max-culpable clauses (the clauses whose width exceeds max)
            guilty_clauses = {k: np.array(embedding[k]) for k in disjunctions if
                              len(embedding[k]) > MAX_WIDTH_THRESHOLD}
            for current_clause, neighbours in guilty_clauses.items():  # 2.2 Split the clauses while keeping planarity
                degree = len(neighbours)
                new_alloc_sizes = np.random.randint(MIN_WIDTH_THRESHOLD, MAX_WIDTH_THRESHOLD + 1,
                                                    (degree//MIN_WIDTH_THRESHOLD) + 1)  # Guarantees split
                new_alloc_indices = np.cumsum(new_alloc_sizes)
                # Split indices computed. Ensure that the last clause is big enough ( >= min_width)
                alloc_indices_cut = new_alloc_indices[new_alloc_indices <= degree - MIN_WIDTH_THRESHOLD]
                pivot = alloc_indices_cut[-1] if len(alloc_indices_cut) > 0 else 0
                if degree - pivot > MAX_WIDTH_THRESHOLD:  # Guarantee that clauses meet the size spec
                    np.append(alloc_indices_cut, np.random.randint(MIN_WIDTH_THRESHOLD,
                                                                   degree - MIN_WIDTH_THRESHOLD - pivot + 1))
                new_sub_neighborhoods = np.split(neighbours, alloc_indices_cut)[1:]
                # 2.3 Create the new disjunctions
                nb_new_disj = len(new_sub_neighborhoods)
                graph.add_nodes_from(["C"+str(nb_disj+idx+1) for idx in range(nb_new_disj)], label="c")  # Add new nodes
                for idx, neighborhood in enumerate(new_sub_neighborhoods):  # Add the new edges and remove old disj edge
                    graph.remove_edges_from([(current_clause, neighbor) for neighbor in neighborhood])  # Remove old
                    graph.add_edges_from([("C"+str(nb_disj+idx+1), neighbor) for neighbor in neighborhood])  # Add new
                nb_disj += nb_new_disj  # Now increment the edges
        # Step 3: If meets the criteria, convert the graph to a formula with negated vars
        # Step 3.1: Rename the nodes and annotate them
        mapping = {label: "C"+str(idx+1) for idx, label in enumerate(disjunctions)}
        mapping.update({label: "V"+str(idx+1) for idx, label in enumerate(variables)})
        label_dict = {x: "c" for x in disjunctions}
        label_dict.update({x: "v" for x in variables})
        nx.classes.set_node_attributes(graph, name="label", values=label_dict)
        graph_sat = nx.relabel_nodes(graph, mapping)
        # Step 3.2: Add the negated variables as nodes. Convention, negated variable is - idx, for simplicity
        graph_sat.add_nodes_from(["V"+str(-idx-1) for idx in range(nb_var)], label="v")  # Add Negated variables
        # 3.3: Randomly assign current/edges to negative variables
        for edge in copy.deepcopy(graph_sat.edges()):  # Since graph is bipartite, guaranteed that only one node is a V
            if edge[0][0] == "V":  # Is this a variable?
                if np.random.uniform() < 0.5:  # 50/50 between positive and negative
                    graph_sat.remove_edge(*edge)
                    graph_sat.add_edge("V-"+edge[0][1:], edge[1])
            elif edge[1][0] == "V":  # Is this a variable?
                if np.random.uniform() < 0.5:
                    graph_sat.remove_edge(*edge)
                    graph_sat.add_edge(edge[0], "V-" + edge[1][1:])

        if not sat_check_from_graph(nb_var, nb_disj, graph_sat):  # Step 4: Check satisfiability
            return None
        # Step 5: Add the positive/negative literal edge connections
        graph_sat.add_edges_from([("V"+str(idx+1), "V-"+str(idx+1)) for idx in range(nb_var)])
        # Step 6: Check whether the introduction of the positive/negative literal division preserved planarity
        pl_check, pl_emb = nx.algorithms.check_planarity(graph_sat)
        if not pl_check:
            return None
        # Step 7: Remove Redundancies in case planar (i.e. disjunctions (conjuncts) with the same in-neighbours)
        pl_emb_conj = {x: str(sorted(y)) for x, y in pl_emb.get_data().items() if x[0] == "C"}  # Disjunctions only
        # 7.1 Flip the dictionary
        neighborhoods = {y: [] for y in pl_emb_conj.values()}  # Get values
        # 7.2: Assemble into list of conjunctions with the same signature
        [(lambda x, y: neighborhoods[y].append(x))(x, y) for x, y in pl_emb_conj.items()]
        # 7.3: Flatten list to identify redundant conjunctions
        disjs_to_remove = [cnjts for dup_list in neighborhoods.values() for cnjts in list_helper(dup_list)]
        nb_disjs_to_remove = len(disjs_to_remove)  # Number of Disjunctions to remove
        if nb_disjs_to_remove > 0:
            graph_sat.remove_nodes_from(disjs_to_remove)  # 7.4 Remove Redundant Disjunctions (i.e. Conjuncts/Clauses)
            # 7.5 Update Names if removal is done ...
            original_set = ["C"+str(x) for x in range(1, nb_disj+1) if "C"+str(x) not in disjs_to_remove]
            post_removal_mapping = {x: "C"+str(idx+1) for idx, x in enumerate(original_set)}
            # post_removal_mapping.update({"V"+str(x): "V"+str(x) for x in range(1, nb_var+1)})
            # post_removal_mapping.update({"V-" + str(x): "V-" + str(x) for x in range(1, nb_var + 1)})
            nb_disj -= nb_disjs_to_remove  # Update the number of disunctions to remove
            graph_sat = nx.relabel_nodes(graph_sat, post_removal_mapping)  # Rename the disjunctions
        # Step 8: Insert SAT/UNSAT cores
        graph_unsat = nx.Graph.copy(graph_sat)  # 6.1 duplicate the graph
        # Mar 3: Lowered max chain length to 4 to lower number of nodes
        chain_length = 2 * np.random.randint(2, 5)  # 6.2 Select a random chain length, when 1 chain is tautology
        sat_core, unsat_core = generate_cores(nb_var + 1, nb_disj + 1, chain_length, draw=False)
        graph_sat.add_nodes_from(sat_core.nodes(data=True))  # 6.3 Inject the elements into the graphs
        graph_sat.add_edges_from(sat_core.edges())
        graph_unsat.add_nodes_from(unsat_core.nodes(data=True))
        graph_unsat.add_edges_from(unsat_core.edges())
        '''# -------- SANITY CHECKS -------- #
        print("Graphs Isomorphic?: "+str(nx.is_isomorphic(graph_sat, graph_unsat)))
        print("SAT Planarity:" + str(nx.algorithms.check_planarity(graph_sat)[0]))
        print("UNSAT Planarity:" + str(nx.algorithms.check_planarity(graph_unsat)[0]))
        print("SAT:" + str(sat_check_from_graph(nb_var + chain_length, nb_disj + 2 * chain_length, graph_sat)))
        print("UNSAT:" + str(sat_check_from_graph(nb_var + chain_length, nb_disj + 2 * chain_length, graph_unsat)))
        print("SAT 1-WL Hash:   " +
              str(nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph_sat, node_attr="label")))
        print("UNSAT 1-WL Hash: " +
              str(nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph_unsat, node_attr="label")))'''
        return graph_sat, graph_unsat
    else:
        return None   # Otherwise, return Nothing


def convert_to_kgnn_format(graph: nx.Graph, label):  # PyTorch Geometric Data
    mapping = {x: idx for idx, x in enumerate(graph.nodes())}  # Convert to integer IDs
    type_mapping = np.expand_dims(np.array([get_node_type(x, return_str=False)
                                            for idx, x in enumerate(graph.nodes())]), axis=-1)
    graph_renamed = nx.relabel_nodes(graph, mapping)  # And apply the renaming
    edges = list(graph_renamed.edges())
    edges.extend([(y, x) for x, y in edges])  # Add every node in the other direction
    edge_indices_np = np.array(edges).T
    data = Data(x=torch.tensor(type_mapping, dtype=torch.long),
                edge_index=torch.tensor(edge_indices_np, dtype=torch.long), y=torch.tensor(label, dtype=torch.long))
    return data


def get_node_type(node_name: str, return_str=True):
    if node_name[0] == "C":
        return "0" if return_str else 0  # Clause
    elif node_name[0] == "V":
        return "1" if return_str else 1  # Vertex
    else:
        raise TypeError("Invalid Node Name/Type")


def convert_to_maron_format(graph: nx.Graph, label: List):
    string_labels = [str(x) for x in label]  # Convert to string array
    mapping = {x: idx for idx, x in enumerate(graph.nodes())}
    type_mapping = {idx: get_node_type(x) for idx, x in enumerate(graph.nodes())}
    graph_renamed = nx.relabel_nodes(graph, mapping)
    neighbour_dict = {n: [str(x) for x, y in nbrdict.items()] for n, nbrdict in graph_renamed.adjacency()}
    degree_dict = {n: str(len(neigh)) for n, neigh in neighbour_dict.items()}
    lines = []  # These are the lines in Maron Format to be returned when outputting the files
    nb_nodes = len(graph_renamed.nodes())
    first_line = str(nb_nodes)+" "+" ".join(string_labels)
    lines.append(first_line)
    node_lines = [type_mapping[x]+" "+degree_dict[x]+" "+" ".join(neighbour_dict[x]) for x in graph_renamed.nodes()]
    lines.extend(node_lines)
    return lines


def parse_plantri_out(target_nb, show_progress):   # Parse Plantri output
    grph_pairs = []  # Graph pairs returned by the processing
    graph_embeddings = []  # The graph planar embeddings
    # While loop statistics
    byte_counter = 1  # The byte counter starting at 1
    nb_read_graphs = 0  # The number of read graphs
    # State information for parser
    state = 0  # The state of the parser
    crt_node_index = 0  # The node index
    crt_node_count = 0  # The node count within graph reading
    crt_graph = nx.Graph()  # Init for graph
    crt_embedding = {}  # Init for graph embedding
    pbar = None  # To avoid the annoying IDE warning
    if show_progress:
        if target_nb is not None and target_nb > 0:  # No Target number of graphs specified, use the whole graph
            pbar = tqdm(total=target_nb)
        else:
            pbar = tqdm(total=os.path.getsize(TEMPORARY_FILE))  # Just to make it nicer
    with open(TEMPORARY_FILE, "rb") as f:
        input_byte = f.read(1)
        while input_byte:  # While loop instead of for loop. Python implementation, so will be slow...
            # Step 1: Only read the line bytes if they exist
            integer_equiv = input_byte[0]
            state, crt_graph, crt_node_index, crt_node_count, save_graph \
                = parse_planar_code(state, integer_equiv, crt_node_index, crt_node_count, crt_graph, crt_embedding)
            if save_graph:
                graph = nx.Graph.copy(crt_graph)
                embedding = copy.deepcopy(crt_embedding)
                graphs = process_graph(graph, embedding)  # Do the normal processing
                if graphs is not None:
                    grph_pairs.append(graphs)
                    graph_embeddings.append(embedding)
                    nb_read_graphs += 1
                    if target_nb is not None and target_nb > 0:
                        if show_progress:
                            pbar.update(1)
                        if nb_read_graphs >= target_nb:
                            break  # End the loop
                crt_graph = nx.Graph()  # Reset the current variables
                crt_embedding = {}
            byte_counter += 1
            input_byte = f.read(1)
            if target_nb is None and show_progress:
                pbar.update(1)
    # print("Total Graphs Processed: "+str(nb_read_graphs))
    os.remove(TEMPORARY_FILE)  # Delete the temporary file
    return grph_pairs  # Can also return embeddings


# CLI with plantri to generate planar BP graphs
def call_plantri(nb_nodes, target_nb: int = None, show_progress=True, timeout=1.5):
    try:
        # Get the outputs saved in a text file, which you can read from later
        subprocess.check_output([PLANTRI_DIR, str(nb_nodes),  # Removed Random Portion non-sense, e.g. 20nodes P0 is 1
                                 "-c1m"+str(MIN_WIDTH_THRESHOLD), "-bp", "-h", TEMPORARY_FILE], timeout=timeout)
        # By setting the min width threshold as min degree, we ensure all clauses meet the width requirement,
        # at the cost of all variables appearing in as many clauses at least
        # In case plantri finishes on time
        return parse_plantri_out(target_nb, show_progress)
    except subprocess.TimeoutExpired:
        return parse_plantri_out(target_nb, show_progress)


# Parser for planar code, useful for getting the graph embeddings as well
def parse_planar_code(state, input_byte, crt_node_index, crt_node_count, crt_graph: nx.Graph, crt_embedding):
    if state == 0:  # 0: Number of nodes not known yet
        nb_nodes = int(input_byte)
        assert nb_nodes > 0
        crt_graph.add_nodes_from(range(nb_nodes))  # Add the nodes to the graph
        crt_embedding.update({x: [] for x in range(nb_nodes)})  # And instantiate the embedding dictionary
        return 1, crt_graph, 0, nb_nodes, False   # New State, Graph, index, count, save graph
    elif state == 1:  # 1: Number of nodes known, reading neighbourhood information
        input_int = int(input_byte)
        input_int_dec = input_int - 1
        # print("Input Byte: " + str(input_int))
        if input_int > 0:
            crt_embedding[crt_node_index].append(input_int_dec)  # Update Embedding Dictionary
            if input_int_dec < crt_node_index:  # So that edges aren't added twice
                crt_graph.add_edge(crt_node_index, input_int_dec)  # Update graph
            return 1, crt_graph, crt_node_index, crt_node_count, False
        elif input_int == 0:   # Separator
            new_crt_index = crt_node_index + 1
            if new_crt_index == crt_node_count:  # End of graph reading
                # print("Graph Read Complete")
                return 0, crt_graph, 0, 0, True  # Reset the state
            else:
                return 1, crt_graph, new_crt_index, crt_node_count, False


def minisat_clause_print(clauses):
    for clause in clauses:
        print(" ".join([str(x) for x in clause]) + " 0")


def graph_to_clauses(graph, nb_disj, norm=True):
    clauses = [[] for _ in range(nb_disj)]  # Build the clause representations
    for edge in nx.edges(graph):  # Build the MiniSAT format formula to be fed into the solver
        if edge[0][0] == "C":
            idx_to_append = int(edge[0][1:]) - 1
            clauses[idx_to_append].append(int(edge[1][1:]))
        elif edge[1][0] == "C":
            idx_to_append = int(edge[1][1:]) - 1
            clauses[idx_to_append].append(int(edge[0][1:]))
    if norm:
        max_length = max([len(x) for x in clauses])
        for x in clauses:
            if len(x) < max_length:
                x.extend([0] * (max_length - len(x)))
    return clauses


def identify_core(graph):  # Cores added last, so they are the highest-indexed variables
    max_var = max([int(x[1:]) for x in graph.nodes if x[0] == "V"])  # Retrieve the literal nodes
    disjs_connecting = [x[1] for x in graph.edges("V"+str(max_var)) if x[1][0] == "C"]  # Get the Conjuncts
    second_deg_vars = [x[1] for x in graph.edges(disjs_connecting) if x[1][0] == "V"]   # And back to the vars
    min_var = min([abs(int(x[1:])) for x in second_deg_vars])  # Get the other side of the core
    return min_var, max_var


# Helper function for CEXP generation: flipping UNSAT graphs to make them SAT by adding edges
def flip_graph(inp_graph, modify_original=True, minimal_addition=True):
    if modify_original:
        graph = inp_graph
    else:
        graph = nx.Graph.copy(inp_graph)
    literals = [x for x in graph.nodes if x[0] == "V"]  # Add edges
    nb_literals = len(literals)
    disjunctions = [x for x in graph.nodes if x[0] == "C"]
    nb_disj = len(disjunctions)
    operations_completed = 0
    still_as_before = True
    min_ops = np.random.randint(3, 7)  # Between 3 and 6 (can go higher due to while loop)
    edges_added = []  # List of edges added
    # Identify the separate components
    while operations_completed < min_ops or still_as_before:
        # Only add to disjunctions less than maximum width
        eligible_disj = [disj for disj in disjunctions if len([x for x in graph.neighbors(disj)]) < MAX_WIDTH_THRESHOLD]
        disj = np.random.randint(len(eligible_disj))   # Pick a disjunction from there
        disjunction = eligible_disj[disj]
        disj_neighbours = [abs(int(x[1:])) for x in graph.neighbors(disjunction)]  # Now find the variables it contains
        eligible_lit = [lit for lit in literals if abs(int(lit[1:])) not in disj_neighbours]
        lit = np.random.randint(len(eligible_lit))  # Choose a random variable. Redundancy now not possible
        graph.add_edge(eligible_lit[lit], disjunction)  # This prevents duplicate edges by default
        edges_added.append((eligible_lit[lit], disjunction))  # Add to the list of added edges
        still_as_before = not sat_check_from_graph(graph=graph, nb_conj=nb_disj, nb_var=(nb_literals // 2))
        operations_completed += 1   # Edge Addition
    edge_removal_candidates = [x for x in edges_added]
    if minimal_addition:
        for edge in edge_removal_candidates:  # Try removing every redundant clause
            g_copy = nx.Graph.copy(graph)  # Duplicate the graph to preserve it
            g_copy.remove_edge(edge[0], edge[1])  # Remove the added disjunction
            if not sat_check_from_graph(nb_var=nb_literals // 2, nb_conj=nb_disj, graph=g_copy):
                continue   # Removing this makes it UNSAT
            else:  # It's still SAT
                graph = g_copy
    if not modify_original:
        return graph


# Distribution: a list of 2-tuples (NbPlantriNodes, Count)
def generate_graphs(distribution, verbose=True, save_maron=False, save_kgnn=False,
                    save_directory=None, name=None, uneven_split=False, flip=False, modulo=4, proportion=1):
    # distribution: A list of 2-tuples (size, number of graph pairs)
    # verbose: Boolean indicating whether the function should print to console
    # save_maron: Save dataset in format usable my the official PPGN codebase
    # save_kgnn: Save dataset in format compatible with the k-GNN code by Morris et al (PyTorch Geometric)
    # name: Dataset name
    # save_directory: where to save the dataset
    # uneven_split: experimental parameter to eliminate some positive examples and make a 2/1 data ratio
    # flip: Used to produce CEXP. by setting flip to True. Some of the unsat examples are converted to SAT,
    # and these replace the original "unflipped" SAT core.
    # proportion and modulo (an even number), how many pairs of graphs do we want to flip?
    # If modulo <= proportion, flip all SAT
    # Otherwise flip all pairs where index % modulo <= proportion
    # E.g. prop 1 mod 6 flips one third of SAT graphs, 1 mod 4 flips 1/2, 3 mod 8 flips half as well,
    # but in a different chunking, etc.
    graph_pairs = {x[0]: [] for x in distribution}
    for tple in distribution:
        if verbose:
            print("Generating "+str(tple[1])+" graphs with "+str(tple[0])+" Plantri nodes")
        graph_pairs[tple[0]] = call_plantri(tple[0], target_nb=tple[1], show_progress=verbose)
    flat_graphs = []
    if save_maron or save_kgnn:   # Two-Tuple, with index 0 --> SAT, 1 --> UNSAT
        for node_nb, graph_pair_set in graph_pairs.items():  # Graph Flattening across all node nbs
            flat_graphs.extend([(graph, [1-idx]) for pair in graph_pair_set for idx, graph in enumerate(pair)])

    if flip:
        temp_flat_graphs = []
        for idx, x in enumerate(flat_graphs):
            if idx % 2 == 0 and idx % modulo <= proportion:  # Even number (SAT), get UNSAT graph and add minimal edges
                temp_flat_graphs.append((flip_graph(inp_graph=flat_graphs[idx + 1][0], modify_original=False,
                                                    minimal_addition=True), [1]))
            else:
                temp_flat_graphs.append(x)  # Keep as is
        flat_graphs = temp_flat_graphs
    if save_maron:   # Convert to Maron format
        nb_graphs = len(flat_graphs)  #
        lines_maron = [convert_to_maron_format(graph=graph[0], label=graph[1]) for graph in flat_graphs]
        # Now need to flatten
        if uneven_split:
            lines_maron = [x for idx, x in enumerate(lines_maron) if idx % 4 !=0]
        lines_maron_flat = [graph_line + "\n" for graph_lines in lines_maron for graph_line in graph_lines]
        lines_maron_flat.insert(0, str(nb_graphs) + "\n")
        if not os.path.exists(save_directory):  # If the destination directory doesn't exist
            os.mkdir(save_directory)
        target_path = os.path.join(save_directory, name)+".txt"
        f = open(target_path, "w")  # Write the file
        f.writelines(lines_maron_flat)
        f.close()
    if save_kgnn:
        graphs_kgnn = [convert_to_kgnn_format(graph=graph[0], label=graph[1]) for graph in flat_graphs]
        if uneven_split:
            graphs_kgnn = [x for idx, x in enumerate(graphs_kgnn) if idx % 4 != 0]
        if not os.path.exists(save_directory):  # If the destination directory doesn't exist
            os.mkdir(save_directory)
        raw_save_directory = os.path.join(save_directory, "raw")
        if not os.path.exists(raw_save_directory):  # If the destination directory doesn't exist
            os.mkdir(raw_save_directory)
        target_path = os.path.join(raw_save_directory, name)+".pkl"
        f = open(target_path, "wb")  # Write the file using pickle
        pickle.dump(graphs_kgnn, f)
        f.close()
    return graph_pairs


if __name__ == "__main__":
    # A Description of the generate_graph function and its options is provided in the function definition
    # EXP
    graph_pairs = generate_graphs([(12, 500), (15, 100)],
                                  save_maron=True, save_directory="Data/",
                                  name="EXP", verbose=True, save_kgnn=True, uneven_split=False, flip=False)
    # CEXP
    graph_pairs_c = generate_graphs([(12, 500), (15, 100)],
                                    save_maron=True, save_directory="Data/", name="CEXP", verbose=True,
                                    save_kgnn=True, uneven_split=False, flip=True, modulo=4, proportion=1)
    # Torch Geometric files are saved in a raw subdirectory under the save directory, while the PPGN format is saved
    # as a txt file
