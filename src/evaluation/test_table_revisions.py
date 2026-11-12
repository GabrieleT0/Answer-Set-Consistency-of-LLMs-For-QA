"""Check baseline immutability and displayed-value revision marking."""
from pathlib import Path
import tempfile
import unittest

from src.evaluation.table_revisions import (
    BLUE, cell_value, compare_table, freeze_base, load_base, strip_marks,
)


def table(value, caption='Accuracy is 1.00.', key='classification'):
    return ('\\begin{table}\n\\caption{' + caption + '}\n'
            '\\begin{tabular}{llllllll}\n'
            'A & Model & ' + value + ' & 2.00 & 3.00 & 4.00 & 5.00 & 6.00 \\\\\n'
            '\\end{tabular}\n\\label{tab:' + key + '}\n\\end{table}\n')


class TableRevisionTests(unittest.TestCase):
    def test_nested_marks_are_removed_without_removing_bold_or_shading(self):
        source = r'\update{\textcolor{blue!60!black}{\textbf{1.00}}} \shadeA{0}'
        self.assertEqual(strip_marks(source), r'\textbf{1.00} \shadeA{0}')
        self.assertEqual(strip_marks(r'\textcolor{red}{1}'), r'\textcolor{red}{1}')

    def test_unchanged_value_ignores_bold_and_existing_blue(self):
        clean, blue, changes = compare_table(table('1.00'), table(BLUE + r'{\textbf{1.00}}'), 'classification')
        self.assertEqual(clean, blue)
        self.assertEqual(changes, [])
        self.assertNotIn(BLUE, blue)

    def test_only_changed_number_is_blue(self):
        clean, blue, changes = compare_table(table('1.00'), table('1.01'), 'classification')
        self.assertIn(BLUE + '{1.01}', blue)
        self.assertNotIn(BLUE + '{2.00}', blue)
        self.assertEqual(len(changes), 1)
        self.assertEqual(strip_marks(blue).split(), clean.split())

    def test_significance_compares_displayed_band(self):
        self.assertEqual(cell_value(r'\shadeA{0.02}', True), cell_value(r'\shadeA{0.03}', True))
        self.assertNotEqual(cell_value(r'\shadeA{0.01}', True), cell_value(r'\shadeA{0.009}', True))
        self.assertEqual(cell_value('--', True), cell_value(r'\shadeA{0.5}', True))

    def test_correlation_stars_and_p_inequality_are_part_of_display(self):
        self.assertNotEqual(cell_value('0.60'), cell_value('0.60*'))
        self.assertNotEqual(cell_value('$<$0.001'), cell_value('0.001'))
        self.assertEqual(cell_value('1.0'), cell_value('1.00'))

    def test_caption_marks_only_changed_tokens_and_preserves_math(self):
        clean, blue, changes = compare_table(table('1.00'), table('1.00', 'Accuracy is 2.00.'), 'classification')
        self.assertIn('Accuracy is ' + BLUE + '{2.00}.', blue)
        self.assertEqual(changes, [])
        self.assertEqual(strip_marks(blue), clean)

    def test_row_changes_require_review(self):
        with self.assertRaisesRegex(ValueError, 'Row identities changed'):
            compare_table(table('1.00'), table('1.00').replace('A & Model', 'B & Model'), 'classification')

    def test_freeze_uses_manuscript_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'main.tex'
            source.write_text(table('1.00'))
            base = root / 'base'
            mapping = {'main': 'tab:classification'}
            freeze_base(base, source, source, mapping)
            self.assertEqual(load_base(base, mapping)['main'], table('1.00'))
            original = (base / 'main.tex').read_bytes()
            source.write_text(table('9.00'))
            with self.assertRaises(FileExistsError):
                freeze_base(base, source, source, mapping)
            self.assertEqual((base / 'main.tex').read_bytes(), original)
            (base / 'main.tex').write_text(table('9.00'))
            with self.assertRaisesRegex(ValueError, 'modified'):
                load_base(base, mapping)


if __name__ == '__main__':
    unittest.main()
