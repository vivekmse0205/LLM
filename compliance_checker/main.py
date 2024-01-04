import argparse
import logging
from src.marketing_compliance import MarketingCompliance

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process(url):
    """
        Process the given URL for marketing compliance and print the results.

        Parameters:
        - url (str): The URL to be processed.
    """
    compliance = MarketingCompliance()
    output = compliance.get_compliance_report(url)
    logger.info(f"Compliance report for URL '{url}': {output}")


def main():
    """
        Main function to test the marketing compliance processing.
    """
    parser = argparse.ArgumentParser(description='Process URL for marketing compliance.')
    parser.add_argument('--url', type=str, default='https://www.joinguava.com/', help='URL to be processed')

    args = parser.parse_args()
    url_to_test = args.url

    if not url_to_test:
        parser.error('Please provide a valid URL using --url.')
    process(url_to_test)


if __name__ == '__main__':
    main()
