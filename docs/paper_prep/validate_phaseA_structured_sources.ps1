param(
    [string]$Manifest = (Join-Path $PSScriptRoot "34a_phaseA_structured_sources.json"),
    [string]$Output = (Join-Path $PSScriptRoot "34a_phaseA_validation.json")
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$tempDir = Join-Path $repoRoot "tmp/pdfs/phaseA_validation"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

function Normalize-Text([string]$Text) {
    return [regex]::Replace($Text.ToLowerInvariant(), "[^\p{L}\p{Nd}]", "")
}

function Test-PdfHeader([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] 5
        $read = $stream.Read($buffer, 0, $buffer.Length)
        return $read -eq 5 -and [Text.Encoding]::ASCII.GetString($buffer) -eq "%PDF-"
    }
    finally {
        $stream.Dispose()
    }
}

$sources = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
$results = foreach ($source in $sources) {
    $pdfPath = Join-Path $repoRoot $source.pdf
    if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
        throw "Missing source PDF: $pdfPath"
    }

    $pageOnePath = Join-Path $tempDir ($source.id + ".page1.txt")
    $fullTextPath = Join-Path $tempDir ($source.id + ".full.txt")
    & pdftotext -f 1 -l 1 $pdfPath $pageOnePath
    if ($LASTEXITCODE -ne 0) { throw "pdftotext page-one extraction failed: $pdfPath" }
    & pdftotext $pdfPath $fullTextPath
    if ($LASTEXITCODE -ne 0) { throw "pdftotext full extraction failed: $pdfPath" }

    $pageOne = Normalize-Text (Get-Content -Raw -LiteralPath $pageOnePath)
    $fullTextRaw = Get-Content -Raw -LiteralPath $fullTextPath
    $fullText = Normalize-Text $fullTextRaw
    $titlePass = $pageOne.Contains((Normalize-Text $source.title))
    $missingAuthors = @($source.authors | Where-Object {
        -not $pageOne.Contains((Normalize-Text $_))
    })
    $authorsPass = $missingAuthors.Count -eq 0
    $quotePass = $fullText.Contains((Normalize-Text $source.quote))
    $hashActual = (Get-FileHash -Algorithm SHA256 -LiteralPath $pdfPath).Hash.ToLowerInvariant()
    $hashPass = $hashActual -ceq $source.sha256
    $headerPass = Test-PdfHeader $pdfPath

    $quotePageHits = @()
    $pages = $fullTextRaw -split [char]12
    for ($i = 0; $i -lt $pages.Count; $i++) {
        if ((Normalize-Text $pages[$i]).Contains((Normalize-Text $source.quote))) {
            $quotePageHits += $i + 1
        }
    }
    $quotePagePass = $quotePageHits -contains [int]$source.quote_pdf_page
    $identityPass = $headerPass -and $titlePass -and $authorsPass
    $allPass = $identityPass -and $quotePass -and $quotePagePass -and $hashPass

    [PSCustomObject]@{
        id = $source.id
        pdf = $source.pdf
        pdf_header_pass = $headerPass
        first_page_title_pass = $titlePass
        first_page_authors_pass = $authorsPass
        missing_first_page_authors = $missingAuthors
        bibliographic_identity_pass = $identityPass
        normalized_quote_pass = $quotePass
        expected_quote_pdf_page = [int]$source.quote_pdf_page
        observed_quote_pdf_pages = $quotePageHits
        quote_page_pass = $quotePagePass
        expected_sha256 = $source.sha256
        actual_sha256 = $hashActual
        full_sha256_pass = $hashPass
        all_three_checks_pass = $allPass
    }
}

$summary = [PSCustomObject]@{
    generated_at = (Get-Date).ToString("o")
    rule = "PDF first-page title and every listed author; normalized full-text quote and page; exact full-file SHA256"
    source_count = $results.Count
    passed_count = @($results | Where-Object all_three_checks_pass).Count
    all_pass = @($results | Where-Object { -not $_.all_three_checks_pass }).Count -eq 0
    results = $results
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
$summary | ConvertTo-Json -Depth 8
if (-not $summary.all_pass) { exit 1 }
