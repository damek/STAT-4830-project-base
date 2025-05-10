import unittest
import random
import torch
import torch.nn as nn

# to run these , append them to the end of the notebook. We will clean this up in the future so we can run these seperately (sorry!)

class TestDataFunctions(unittest.TestCase):

    def test_generate_sequence(self):
        """Test that generate_sequence returns a sequence of the correct length and valid tokens."""
        seq_length = 10
        seq = generate_sequence(seq_length)
        self.assertEqual(len(seq), seq_length)
        # Check that every token is one of the allowed bases (excluding the mask token)
        for token in seq:
            self.assertIn(token, vocab[:-1], msg=f"Token {token} not in allowed vocabulary {vocab[:-1]}.")

    def test_mask_sequence_all_masked(self):
        """Test that with mask probability 1.0 all tokens are masked and labels are correct."""
        seq = ['A', 'C', 'G', 'T']
        masked_seq, labels = mask_sequence(seq, mask_prob=1.0)
        self.assertTrue(all(token == mask_token for token in masked_seq),
                        msg="All tokens should be masked when mask_prob=1.0")
        expected_labels = [vocab_to_idx[t] for t in seq]
        self.assertEqual(labels, expected_labels)

    def test_mask_sequence_none_masked(self):
        """Test that with mask probability 0.0 no tokens are masked and labels are all -100."""
        seq = ['A', 'C', 'G', 'T']
        masked_seq, labels = mask_sequence(seq, mask_prob=0.0)
        self.assertEqual(masked_seq, seq, msg="No tokens should be masked when mask_prob=0.0")
        self.assertEqual(labels, [-100, -100, -100, -100], msg="Labels should be -100 when no tokens are masked")

    def test_process_sequence_valid(self):
        """Test that process_sequence correctly processes a Biopython SeqRecord."""
        # Create a dummy SeqRecord using Biopython
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        # Create a sequence long enough (e.g., 120 characters, when max_seq_len is 100)
        seq_str = "ACGT" * 30  # 120 characters
        record = SeqRecord(Seq(seq_str), id="test")
        result = process_sequence(record, max_seq_len=100, mask_prob=0.5)
        self.assertIsNotNone(result, msg="process_sequence should return a result for sequences longer than max_seq_len")
        tokens, masked_tokens, labels = result
        self.assertEqual(len(tokens), 100)
        self.assertEqual(len(masked_tokens), 100)
        self.assertEqual(len(labels), 100)
        # For each position where the label is not -100, the token should be the mask token and label valid.
        for token, label in zip(masked_tokens, labels):
            if label != -100:
                self.assertEqual(token, mask_token)
                self.assertIn(label, list(vocab_to_idx.values()))

    def test_dnaseq_dataset(self):
        """Test that the DNADataset returns tensors with correct dimensions."""
        dataset = DNADataset(data, vocab_to_idx)
        # Get the first data point
        input_ids, labels = dataset[0]
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(labels, torch.Tensor))
        self.assertEqual(input_ids.dim(), 1)
        self.assertEqual(labels.dim(), 1)

class TestTransformerModel(unittest.TestCase):

    def test_forward_output_shape(self):
        """Test that TransformerModel produces output of expected shape."""
        batch_size = 4
        seq_len = 10
        input_tensor = torch.randint(0, len(vocab_to_idx), (batch_size, seq_len)).to(device)
        outputs = model(input_tensor)  # model should be an instance of TransformerModel
        # Expected output shape: [batch_size, seq_len, vocab_size]
        self.assertEqual(outputs.shape, (batch_size, seq_len, len(vocab_to_idx)),
                         msg="TransformerModel output shape is incorrect.")

class TestRealDNADataset(unittest.TestCase):

    def test_dataset_item(self):
        """Test that RealDNADataset returns correctly processed tensor items."""
        # Only run this test if there is processed data available.
        if len(processed_data) == 0:
            self.skipTest("No processed_data available for testing RealDNADataset.")
        dataset = RealDNADataset(processed_data, vocab_to_idx)
        input_ids, labels = dataset[0]
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(labels, torch.Tensor))
        # The length of each sample should match max_seq_len
        self.assertEqual(len(input_ids), max_seq_len)
        self.assertEqual(len(labels), max_seq_len)
s
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)