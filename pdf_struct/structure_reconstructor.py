import copy
import itertools
from typing import List, Set, Optional, Dict, NamedTuple

import numpy as np

from pdf_struct.listing import SectionNumber
from pdf_struct.transition_labels import ListAction, DocumentWithFeatures


class Paragraph(NamedTuple):
    text: str
    level: int


class MultiLevelListingBlock(object):
    def __init__(self, list_candidates: Set[SectionNumber], text: str):
        # list style candidates at start of each level
        # when len(self._level_start_listing_candidates[i]) == 1, that means
        # that layer i has a consolidated listing style
        # We keep candidates even after ListAction.DOWN because it will be
        # used for matching
        self._level_start_listing_candidates: List[Set[SectionNumber]] = [copy.deepcopy(list_candidates)]
        # list style candidates at end of each level
        self._level_last_listing_candidates: List[Set[SectionNumber]] = [copy.deepcopy(list_candidates)]
        self._texts: List[List[str]] = [[text]]
        self._levels: List[int] = [0]
        # maps number to each level
        # This is redundant to self._level_* but is here for performance issue
        # it will created by self.index_level command
        self._listing_to_level: Optional[Dict[SectionNumber, int]] = None

    def listing_to_level(self, section_number: SectionNumber) -> int:
        return self._listing_to_level[section_number]

    def append(self, list_candidates: Set[SectionNumber], text: str, action: ListAction):
        if action == ListAction.CONTINUOUS:
            self._texts[-1].append(text)
        elif action == ListAction.SAME_LEVEL:
            self._texts.append([text])
            self._levels.append(len(self._level_last_listing_candidates) - 1)
            if len(list_candidates) == 0:
                # non-list paragraph can come after any listing style, so
                # do not add it to self._level_last_listing_candidates
                return
            if len(self._level_last_listing_candidates[-1]) == 1:
                # Already a consolidated level ... either update the section
                # or ignore the input candidates that are probably false positives
                section_number_pre = self._level_last_listing_candidates[-1].pop()
                for section_number in list_candidates:
                    if section_number.is_next_of(section_number_pre):
                        self._level_last_listing_candidates[-1].add(section_number)
                        return
                # no valid update found -> just keep the current number
                self._level_last_listing_candidates[-1].add(section_number_pre)
            # listing style for current level is not consolidated
            for section_number_pre, section_number in itertools.product(self._level_last_listing_candidates[-1], list_candidates):
                if section_number.is_next_of(section_number_pre):
                    # Proper consecutive listing found -> trust this candidate
                    self._level_last_listing_candidates[-1] = {section_number}
                    # Since we trust the sequence if we find at least one consecutive
                    # listing, the first section number of the level must be
                    # section_number_pre.
                    self._level_start_listing_candidates[-1] = {section_number_pre}
                    return
            # No consecutive sections found: just add them to candidate
            self._level_last_listing_candidates[-1].update(list_candidates)
            # it could be that existing listing candidates are all false positives and section
            # starts from the current section_number, add them to the start as well
            self._level_start_listing_candidates[-1].update(list_candidates)
        elif action == ListAction.UP:
            raise ValueError('MultiLevelListingBlock.append does not expect ListAction.UP')
        elif action == ListAction.DOWN:
            self._texts.append([text])
            self._levels.append(len(self._level_last_listing_candidates))
            self._level_start_listing_candidates.append(copy.deepcopy(list_candidates))
            self._level_last_listing_candidates.append(copy.deepcopy(list_candidates))
        elif action == ListAction.UP:
            raise ValueError('MultiLevelListingBlock.append does not expect ListAction.ELIMINATE')
        else:
            assert not 'Should not reach here'

    def index_level(self):
        # If two levels had exact same number, we prioritize higher level.
        # thus we start from the back (the lower) and update it in higher levels.
        self._listing_to_level = dict()
        for l, section_numbers in list(enumerate(self._level_start_listing_candidates))[::-1]:
            for section_number in section_numbers:
                self._listing_to_level[section_number] = l

    @staticmethod
    def get_levels(blocks: List['MultiLevelListingBlock']) -> List[int]:
        blocks = copy.deepcopy(blocks)
        for block in blocks:
            block.index_level()

        # First, connected blocks
        max_n_candidates = max(
            (len(b._level_last_listing_candidates) for b in blocks))
        # edge_mask[i_b, i_c, j_b, j_c] denotes if we are allowed to have
        # edge from blocks[i_b]._level_last_listing_candidates[i_c] to
        # blocks[j_b]._level_last_listing_candidates[j_c]
        edge_mask = np.ones((len(blocks), max_n_candidates, len(blocks), max_n_candidates))
        edges = np.zeros((len(blocks), max_n_candidates, len(blocks), max_n_candidates))
        for i_b in range(len(blocks) - 1):
            for i_c, cands_i in enumerate(blocks[i_b]._level_last_listing_candidates):
                for j_b in range(i_b + 1, len(blocks)):
                    for j_c, cands_j in enumerate(blocks[j_b]._level_start_listing_candidates):
                        if edge_mask[i_b, i_c, j_b, j_c] == 0:
                            continue
                        for cand_i, cand_j in itertools.product(cands_i, cands_j):
                            if cand_j.is_next_of(cand_i):
                                # consolidate candidates
                                blocks[i_b]._level_last_listing_candidates[i_c] = {cand_i}
                                blocks[j_b]._level_start_listing_candidates[j_c] = {cand_j}
                                edges[i_b, i_c, j_b, j_c] = 1
                                edge_mask[i_b, i_c:, j_b, :j_c+1] = 0
                                edge_mask[i_b, :i_c+1, j_b, j_c:] = 0
                                mask_rooted = np.any(edges[:i_b+1, :, i_b:j_b], axis=(0, 1))
                                edge_mask[i_b:j_b, :, j_b, j_c:][mask_rooted] = 0
                                break  # goto next i_c
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break

        # Use the graph (connected blocks) to calculate global level of each
        # level of each block
        levels = np.full((len(blocks), max_n_candidates), np.nan)
        levels[:, 0] = 0

        # propagete adjency matrix; max propagation occurs when it spans
        # from the first block to the last block (maximum propagation of len(blocks)
        adj_base = edges.copy().reshape(len(blocks) * max_n_candidates, len(blocks) * max_n_candidates)
        adj_base += adj_base.transpose()
        adj_base = adj_base.astype(bool) + np.eye(adj_base.shape[0], dtype=bool)
        adj = np.linalg.matrix_power(adj_base, len(blocks))
        assert np.all(np.matmul(adj, adj_base) == adj) and 'no more propagation occurs'
        l = 0
        while True:
            levels_old = levels.copy()
            mask_prop = np.matmul((levels == l).reshape(1, -1), adj).reshape(levels.shape)
            levels[np.logical_and(mask_prop, np.logical_or(levels < l, np.isnan(levels)))] = l
            # Use np.where instead of np.maximum to overwrite NaN
            levels[:, 1:] = np.where(
                levels[:, :-1] + 1 <= levels[:, 1:],
                levels[:, 1:],
                levels[:, :-1] + 1  # chosen when nan
            )
            if np.all(np.logical_or(levels_old == levels, np.isnan(levels))):
                if l == (max_n_candidates - 1):
                    break
                else:
                    l += 1
                    continue

        def get_crossing_edges(block_ind):
            # Return if there exists any crossing edges at block_ind
            # e.g. if get_crossing_edges(1) -> [0], then it means that
            # there exists an edge from i-th (i < 1) to j-th (1 < j) block
            # and edge connects two blocks at 0-th level
            mask_0 = np.zeros((len(blocks), 1, len(blocks), 1))
            mask_0[:block_ind, ...] = 1
            mask_1 = np.zeros((len(blocks), 1, len(blocks), 1))
            mask_1[:, :, block_ind:] = 1
            masked_edges = edges * mask_0 * mask_1
            # identify if each node of the levels is used in masked_edges
            # -> (len(blocks) * max_n_candidates, )
            used_sources = np.any(
                masked_edges.reshape(masked_edges.shape[0] * masked_edges.shape[1], -1),
                axis=1)
            return np.array(sorted(levels.flat[used_sources]))

        # deal with dangling blocks
        # it is dangling if it does not have a incoming edge to the first level
        for i in np.nonzero(np.logical_not(np.any(edges[:, :, :, 0], axis=(0, 1))))[0]:
            if i == 0:
                continue
            cross_edge_levels = get_crossing_edges(i)
            block_size = len(blocks[i]._level_last_listing_candidates)
            if np.any(edges[:, :, i]) or np.any(edges[i, :, :]):
                top_idx = int(min(
                    np.argmin(np.sum(edges[:, :, i], axis=(0, 1))),
                    np.argmin(np.sum(edges[i, :, :], axis=(0, 1)))))
                top_level = levels[i, top_idx]
                bottom_crossings = cross_edge_levels[cross_edge_levels < top_level]
                if len(bottom_crossings) > 0:
                    bottom_level = np.max(bottom_crossings)
                else:
                    bottom_level = 0
                if (top_level - bottom_level + 1) < top_idx:
                    # create new levels
                    levels[levels >= top_level] += (bottom_level + block_size) - top_level
                levels[i, :top_idx] = np.arange(top_idx) + bottom_level
            else:
                # This is an orphan block that has no incoming/outgoing edge
                last_level = levels[i - 1, len(blocks[i - 1]._level_last_listing_candidates) - 1]
                bottom_crossings = cross_edge_levels[cross_edge_levels < last_level]
                if len(bottom_crossings) > 0:
                    bottom_level = np.max(bottom_crossings)
                else:
                    bottom_level = 0
                upper_crossings = cross_edge_levels[cross_edge_levels >= last_level]
                if len(upper_crossings) > 0:
                    top_level = int(min(upper_crossings))
                else:
                    top_level = last_level
                if (top_level - bottom_level + 1) < block_size:
                    # create new levels
                    levels[levels >= top_level] += (bottom_level + block_size) - top_level
                levels[i, :block_size] = np.arange(block_size) + bottom_level
        return sum([block._global_levels(levels_block)
                    for levels_block, block in zip(levels, blocks)], [])

    def _global_levels(self, levels: List[int]) -> List[int]:
        global_levels = []
        for l, t in zip(self._levels, self._texts):
            global_levels.extend([int(levels[l])] * len(t))
        return global_levels


def construct_hierarchy(document: DocumentWithFeatures,
                        actions: List[ListAction]) -> List[int]:
    blocks = []
    cur_block = None
    # bypass next_action because we want to ignore ListAction.ELIMINATE
    next_action = None
    for yi, text in zip(actions, document.texts):
        if yi == ListAction.ELIMINATE:
            continue  # Keep next_action
        if next_action is None:
            cur_block = MultiLevelListingBlock(
                SectionNumber.extract_section_number(text),
                text.replace('\n', ' ')
            )
        elif next_action == ListAction.UP:
            blocks.append(cur_block)
            cur_block = MultiLevelListingBlock(
                SectionNumber.extract_section_number(text),
                text.replace('\n', ' ')
            )
        else:
            cur_block.append(
                SectionNumber.extract_section_number(text),
                text.replace('\n', ' '), next_action)
        next_action = yi
    blocks.append(cur_block)
    return MultiLevelListingBlock.get_levels(blocks)
