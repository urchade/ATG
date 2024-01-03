import torch
from metric import compute_prf
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, loader, decoding_function=None):
        self.model = model
        self.loader = loader

        self.decoding_function = decoding_function

    def evaluate(self):
        return self.evaluate_all_with_loader(self.model, self.loader)

    @staticmethod
    def get_entities(output_seq):
        all_ents = []
        for i in output_seq:
            if len(i) == 3:
                s, e, lab = i
                if [lab, (s, e)] in all_ents:
                    continue
                all_ents.append([lab, (s, e)])
        return all_ents

    @staticmethod
    def get_relations(dec_i, symetric, exclude_type):
        relations = []

        if dec_i[-1] == "stop_entity":
            return relations

        index_end = dec_i.index("stop_entity")

        for i in range(index_end + 1, len(dec_i), 3):
            head, tail, r_type = dec_i[i:i + 3]

            if exclude_type:
                head = head[0], head[1]
                tail = tail[0], tail[1]

            if symetric or r_type in ["COMPARE", "CONJUNCTION"]:  # sort the head and tail by start index
                if head[0] > tail[0]:
                    head, tail = tail, head

            if head != tail and [r_type, (head, tail)] not in relations:
                relations.append([r_type, (head, tail)])

        return relations

    def extract_entities_and_relations(self, input_seq, symetric, exclude_type):
        try:
            relations_triples = self.get_relations(input_seq, symetric, exclude_type)
        except:
            relations_triples = []
        entities = self.get_entities(input_seq)

        return {
            "entities": entities,
            "relations_triples": relations_triples
        }

    def transform_data(self, all_true, all_outs, symetric=False, exclude_type=False):
        # extract entities and relations
        all_true_ent = []
        all_true_rel = []
        all_outs_ent = []
        all_outs_rel = []
        for i, j in zip(all_true, all_outs):
            e, r = self.extract_entities_and_relations(i, symetric=symetric, exclude_type=exclude_type).values()
            all_true_ent.append(e)
            all_true_rel.append(r)

            e, r = self.extract_entities_and_relations(j, symetric=symetric, exclude_type=exclude_type).values()
            all_outs_ent.append(e)
            all_outs_rel.append(r)

        return all_true_ent, all_true_rel, all_outs_ent, all_outs_rel

    @torch.no_grad()
    def generate(self, model, loader):

        model.eval()
        all_outs = []
        all_true = []
        device = next(model.parameters()).device
        for x in tqdm(loader, desc="Decoding"):
            # Move input tensors to the device
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            if self.decoding_function is None:
                out = model.decode_batch(x)
            else:
                out = self.decoding_function(model, x)
            all_outs.extend(out)
            all_true.extend(x["graph"])

        return all_true, all_outs

    def evaluate_all_combinations(self, all_true, all_outs):

        all_symetric = [False, True]
        all_exclude_type = [False, True]

        output = {}

        for symetric in all_symetric:
            for exclude_type in all_exclude_type:
                all_true_ent, all_true_rel, all_outs_ent, all_outs_rel = self.transform_data(all_true, all_outs,
                                                                                             symetric=symetric,
                                                                                             exclude_type=exclude_type)
                ent_eval = compute_prf(all_true_ent, all_outs_ent)
                rel_eval = compute_prf(all_true_rel, all_outs_rel)

                if exclude_type and symetric:
                    name = "Relaxed + Symetric"
                elif exclude_type and not symetric:
                    name = "Relaxed + not Symetric"
                elif not exclude_type and symetric:
                    name = "Strict + Symetric"
                else:
                    name = "Strict + not Symetric"
                output[f"{name}"] = rel_eval

        output["Entity"] = ent_eval

        # beautiful output string (aligned, formatted and with newlines)
        output_str = ""
        for k, v in output.items():
            precision, recall, f1 = v.values()
            output_str += f"{k}:\n"
            # percentage
            output_str += f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"

        return output, output_str

    def evaluate_all_with_loader(self, model, loader):
        all_true, all_outs = self.generate(model, loader)
        return self.evaluate_all_combinations(all_true, all_outs)
