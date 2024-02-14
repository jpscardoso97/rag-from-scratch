from inference import Inference

def test_search_rag():
    inf = Inference("gpt2")
    input = [{"role": "user", "content": "How much is the tuition?"}]
    result = inf.search_rag(input)
    expected = "program, assuming normal time to completion. There may be additional costs for living expenses. Note that this information is subject to change:  Item Amount  Tuition $8,043 per 3-credit class, and a total of $32,172 for the certificate **  Transcript Fee $120 **  Books $310  TOTAL $32,602  Notes ** Duke may change its tuition for each academic year, and this estimate is based on current academic year tuition charges   Student Loans Information about average student debt, monthly debt expense and debt interest rate have been withheld since fewer than 10 students have completed this program Job Placement Rates We are not currently required to calculate a job placement rate for program completers Program graduates are employed in the "
    assert result == expected
