import pandas as pd
import re
import json
import argparse


def convert_to_json(val):
    if pd.isna(val) or val == 'NaN':
        return None

    text = str(val)

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    try:
        import ast
        obj = ast.literal_eval(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    try:
        text = text.replace("\\'", "__ESCAPED_APOS__")
        text = text.replace("'", '"')
        text = text.replace("__ESCAPED_APOS__", "'")
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            text = text[start:end]
            text = text.replace("'", '"')
            text = re.sub(r'\bTrue\b', 'true', text)
            text = re.sub(r'\bFalse\b', 'false', text)
            text = re.sub(r'\bNone\b', 'null', text)
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Fix malformed JSON in xlsx columns")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--cols", nargs="+", default=["plan_new", "fun_call_new"],
                        help="Columns to fix")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_excel(args.input)
    print(f"Original rows: {len(df)}")

    rows_to_delete = set()
    for col in args.cols:
        if col in df.columns:
            fixed = 0
            failed = 0
            for idx, val in df[col].items():
                result = convert_to_json(val)
                if result:
                    df.at[idx, col] = result
                    fixed += 1
                else:
                    failed += 1
                    rows_to_delete.add(idx)
            print(f"  {col}: fixed={fixed}, failed={failed}")

    for idx in sorted(rows_to_delete, reverse=True):
        df = df.drop(index=idx)

    output = args.output or args.input.replace('.xlsx', '_fixed.xlsx')
    df.to_excel(output, index=False)
    print(f"Saved: {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
