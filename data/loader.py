import pdfplumber
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple

class PDFLoader:
    """Handle PDF loading and text extraction — both essay and numeric results"""

    COURSE_PREFIXES = "EECS|MATH|CSSD|PHYS|ESSE|DIGT|TECL|CCCC|CIVL|ENG|MECH|TRON"

    # Questions to keep (Hand picked during meetings with supervisor)
    KEEP_QUESTIONS = {
        'core':        {1, 4, 6},
        'course':      {4},
        'lect':        {1, 3, 7},
    }

    # Threshold: keep answer rows that together account for this 75% of responses
    CUMULATIVE_THRESHOLD = 0.75   

    def __init__(self):
        self.rows = []

    def extract_essay_metadata(self, block_text: str) -> Tuple:
        """Extract metadata from an ESSAY RESULTS block"""
        course_code  = None
        academic_year = None
        section      = None

        course_match  = re.search(
            rf"Essay Results for:\s+.*?((?:{self.COURSE_PREFIXES})\s+\d{{4}})", block_text
        )
        year_match    = re.search(r"Academic Year:\s+(\d+)", block_text)
        section_match = re.search(r"Section\(s\):\s+([A-Z, ]+)", block_text)

        if course_match:
            course_code   = course_match.group(1)
        if year_match:
            academic_year = year_match.group(1)
        if section_match:
            section       = section_match.group(1).strip()

        return course_code, academic_year, section

    def extract_numeric_metadata(self, block_text: str) -> Tuple:
        """Extract metadata from a NUMERIC RESULTS block"""
        course_code   = None
        academic_year = None
        section       = None

        course_match  = re.search(
            rf"Numeric Results for:\s+.*?((?:{self.COURSE_PREFIXES})\s+\d{{4}})", block_text
        )
        year_match    = re.search(r"Academic Year:\s+(\d+)", block_text)
        section_match = re.search(r"Section\(s\):\s+([A-Z, ]+)", block_text)

        if course_match:
            course_code   = course_match.group(1)
        if year_match:
            academic_year = year_match.group(1)
        if section_match:
            section       = section_match.group(1).strip()

        return course_code, academic_year, section


    def load_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Extract essay reviews → DataFrame"""
        rows = []

        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]

        full_text    = "\n".join(pages_text)
        essay_blocks = full_text.split("ESSAY RESULTS")

        for block in essay_blocks[1:]:
            course_code, academic_year, section = self.extract_essay_metadata(block)

            question_matches = list(re.finditer(r'(\d+\)\s+.+?:)', block))

            for i, q_match in enumerate(question_matches):
                current_question = q_match.group(1)
                start_pos        = q_match.end()
                end_pos          = (
                    question_matches[i + 1].start()
                    if i + 1 < len(question_matches)
                    else len(block)
                )
                question_text = block[start_pos:end_pos]

                bullet_pattern = r'-\s*(.*?)(?=\s*\n\s*-|\s*\n\s*\d+\)|$)'
                bullets        = re.findall(bullet_pattern, question_text, re.DOTALL)

                for bullet in bullets:
                    review_text = re.sub(r'\s+', ' ', bullet.strip())
                    if review_text and review_text != ".":
                        rows.append({
                            "course_code":   course_code,
                            "academic_year": academic_year,
                            "section":       section,
                            "question_text": current_question,
                            "review":        review_text,
                        })

        return pd.DataFrame(rows)

    def load_numeric_pdf(self, pdf_path: str) -> pd.DataFrame:
        
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]

        full_text      = "\n".join(pages_text)
        numeric_blocks = full_text.split("NUMERIC RESULTS")

        all_rows = []
        for block in numeric_blocks[1:]:
            # Stop at ESSAY RESULTS so we don't bleed into the next section
            block = block.split("ESSAY RESULTS")[0]

            course_code, academic_year, section = self.extract_numeric_metadata(block)
            if not course_code:
                continue

            # Split into the three subsections
            subsection_rows = self._parse_numeric_block(
                block, course_code, academic_year, section
            )
            all_rows.extend(subsection_rows)

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df

        df = self._apply_threshold_filter(df)

        return df.reset_index(drop=True)

    def _parse_numeric_block(
        self, block: str, course_code: str, academic_year: str, section: str
    ) -> List[Dict]:
       
        rows = []

        core_match   = re.search(r'Evaluation of Core Institutional Questions', block)
        course_match = re.search(r'Evaluation of Course Level Questions',       block)
        lect_match   = re.search(r'Evaluation of LECT \d+',                    block)

        if not core_match:
            return rows   

        core_start   = core_match.end()
        course_start = course_match.start() if course_match else len(block)
        lect_start   = lect_match.start()   if lect_match   else len(block)

        subsections = {
            'core':   block[core_start:course_start],
            'course': block[course_start:lect_start] if course_match else '',
            'lect':   block[lect_start:] if lect_match else '',
        }

        for sub_key, sub_text in subsections.items():
            wanted = self.KEEP_QUESTIONS.get(sub_key, set())
            if not wanted or not sub_text.strip():
                continue

            parsed = self._parse_subsection(
                sub_text, sub_key, wanted, course_code, academic_year, section
            )
            rows.extend(parsed)

        return rows


    def _parse_subsection(
        self,
        text:         str,
        subsection:   str,
        wanted_qs:    set,
        course_code:  str,
        academic_year: str,
        section:      str,
    ) -> List[Dict]:
        """
        Parse individual questions from a subsection block.
        Only questions whose 1-based index is in wanted_qs are processed.
        """
        rows = []

        q_pattern = re.compile(
            r'(?:^|\n)(\d+)\.\s+(.*?)(?=\n\d+\.\s+|\Z)', re.DOTALL
        )
        matches   = list(q_pattern.finditer('\n' + text))

        for m in matches:
            q_num  = int(m.group(1))
            if q_num not in wanted_qs:
                continue

            q_body = m.group(2)
            q_text_match = re.match(r'(.*?)\n\s*Answers', q_body, re.DOTALL)
            q_text = (
                re.sub(r'\s+', ' ', q_text_match.group(1)).strip()
                if q_text_match
                else ''
            )

            
            mean   = self._extract_stat(q_body, 'Mean')
            median = self._extract_stat(q_body, 'Median')

            rr_match = re.search(r'Response Rate:\s*(\d+)\s*/\s*(\d+)', q_body)
            resp_n     = int(rr_match.group(1)) if rr_match else None
            resp_total = int(rr_match.group(2)) if rr_match else None
            answer_pattern = re.compile(
                r'^(\d)\s+=\s+(.+?)\s+(\d+)\s+([\d.]+)%',
                re.MULTILINE
            )
            for a_match in answer_pattern.finditer(q_body):
                answer_value = int(a_match.group(1))
                answer_label = a_match.group(2).strip()
                frequency    = int(a_match.group(3))
                percentage   = float(a_match.group(4))

                rows.append({
                    'course_code':    course_code,
                    'academic_year':  academic_year,
                    'section':        section,
                    'subsection':     subsection,
                    'question_number': q_num,
                    'question_text':  q_text,
                    'answer_label':   answer_label,
                    'answer_value':   answer_value,
                    'frequency':      frequency,
                    'percentage':     percentage,
                    'mean':           mean,
                    'median':         median,
                    'response_n':     resp_n,
                    'response_total': resp_total,
                })

        return rows


    @staticmethod
    def _extract_stat(text: str, stat_name: str) -> Optional[float]:
        """Extract a named stat like 'Mean: 5.78' from a block of text"""
        m = re.search(rf'{stat_name}:\s*([\d.]+)', text)
        return float(m.group(1)) if m else None


    def _apply_threshold_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Within each (course, year, section, subsection, question) group,
        keep only the answer rows that cumulatively account for
        CUMULATIVE_THRESHOLD of all responses, working from highest
        frequency downward.

        Answer value 0 (Not Applicable) is always dropped first as it
        is not part of the opinion scale.
        """
        # Drop "Not Applicable" rows — they're not opinion data
        df = df[df['answer_value'] != 0].copy()

        kept = []
        group_cols = ['course_code', 'academic_year', 'section',
                      'subsection', 'question_number']

        for _, grp in df.groupby(group_cols, sort=False):
            grp_sorted = grp.sort_values('frequency', ascending=False)
            total      = grp_sorted['frequency'].sum()

            if total == 0:
                continue

            cumulative = 0.0
            for _, row in grp_sorted.iterrows():
                cumulative += row['frequency']
                kept.append(row)
                if cumulative / total >= self.CUMULATIVE_THRESHOLD:
                    break

        return pd.DataFrame(kept)
