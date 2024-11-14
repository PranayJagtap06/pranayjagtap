// search-dropdown.js
class SearchDropdown {
    constructor() {
        this.portfolios = [];
        this.tags = new Set();
        this.init();
    }

    async init() {
        try {
            const response = await fetch('portfolios/portfolios.json');
            this.portfolios = await response.json();

            this.portfolios.forEach(portfolio => {
                portfolio.tags.forEach(tag => this.tags.add(tag));
            });

            this.setupSearchDropdown();
        } catch (error) {
            console.error('Error initializing search dropdown:', error);
        }
    }

    setupSearchDropdown() {
        const searchContainer = document.querySelector('.search-container');
        searchContainer.innerHTML = `
            <div class="search-wrapper">
                <input type="text" class="search-input" id="search-input" placeholder="🔍 Search portfolios or tags...">
                <div class="search-dropdown" id="search-dropdown">
                    <div class="dropdown-section">
                        <div class="section-title">Tags</div>
                        <div class="tags-cloud" id="tags-container"></div>
                    </div>
                    <div class="dropdown-section">
                        <div class="section-title">Portfolios</div>
                        <ul class="portfolios-list" id="portfolios-container"></ul>
                    </div>
                </div>
            </div>
        `;

        this.searchInput = document.getElementById('search-input');
        this.dropdown = document.getElementById('search-dropdown');
        this.tagsContainer = document.getElementById('tags-container');
        this.portfoliosContainer = document.getElementById('portfolios-container');

        // Add input event listener
        this.searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value;
            this.handleSearch(searchTerm);
        });

        this.searchInput.addEventListener('focus', () => this.showDropdown());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const searchTerm = this.searchInput.value;
                if (window.portfolioManager) {
                    window.portfolioManager.searchPortfolios(searchTerm);
                }
                this.hideDropdown();
            }
        });

        document.addEventListener('click', (e) => this.handleClickOutside(e));

        // Check for tag parameter in URL on load
        const urlParams = new URLSearchParams(window.location.search);
        const tagParam = urlParams.get('tag');
        if (tagParam) {
            this.searchInput.value = tagParam;
            if (window.portfolioManager) {
                window.portfolioManager.searchPortfolios(tagParam);
            }
        }

        // Initial render
        this.renderDropdownContent();
    }

    handleSearch(searchTerm) {
        const filteredTags = Array.from(this.tags)
            .filter(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));

        const filteredPortfolios = this.portfolios
            .filter(portfolio =>
                portfolio.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                portfolio.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
            );

        this.renderDropdownContent(filteredTags, filteredPortfolios, searchTerm);

        if (searchTerm && (filteredTags.length || filteredPortfolios.length)) {
            this.showDropdown();
        } else if (!searchTerm) {
            this.hideDropdown();
        }
    }

    renderDropdownContent(filteredTags = Array.from(this.tags), filteredPortfolios = this.portfolios, searchTerm = '') {
        this.tagsContainer.innerHTML = filteredTags
            .map(tag => {
                const highlightedTag = this.highlightMatch(tag, searchTerm);
                return `
                    <div class="dropdown-tag" data-tag="${tag}">
                        <i class="fa-solid fa-tag"></i> ${highlightedTag}
                    </div>
                `;
            })
            .join('');

        this.portfoliosContainer.innerHTML = filteredPortfolios
            .map(portfolio => {
                const highlightedTitle = this.highlightMatch(portfolio.title, searchTerm);
                return `
                    <li class="portfolio-item" data-title="${portfolio.title}">
                        <i class="fa-solid fa-folder"></i>
                        ${highlightedTitle}
                    </li>
                `;
            })
            .join('');

        this.addClickEvents();
    }

    addClickEvents() {
        this.tagsContainer.querySelectorAll('.dropdown-tag').forEach(tagElement => {
            tagElement.addEventListener('click', (e) => {
                e.stopPropagation();
                const tag = tagElement.dataset.tag;
                this.searchInput.value = tag;

                // Update URL
                const urlParams = new URLSearchParams(window.location.search);
                urlParams.set('tag', tag);
                window.history.replaceState({}, '', `${window.location.pathname}?${urlParams}`);

                // Trigger searches
                if (window.portfolioManager) {
                    window.portfolioManager.searchPortfolios(tag);
                }

                this.hideDropdown();
            });
        });

        this.portfoliosContainer.querySelectorAll('.portfolio-item').forEach(itemElement => {
            itemElement.addEventListener('click', (e) => {
                e.stopPropagation();
                const title = itemElement.dataset.title;
                this.searchInput.value = title;

                // Trigger search for the selected portfolio title
                if (window.portfolioManager) {
                    window.portfolioManager.searchPortfolios(title);
                }

                this.hideDropdown();
            });
        });
    }

    highlightMatch(text, searchTerm) {
        if (!searchTerm) return text;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<span class="highlight">$1</span>');
    }

    showDropdown() {
        this.dropdown.classList.add('active');
    }

    hideDropdown() {
        this.dropdown.classList.remove('active');
    }

    handleClickOutside(event) {
        if (!this.dropdown.contains(event.target) && !this.searchInput.contains(event.target)) {
            this.hideDropdown();
        }
    }
}

// Initialize the search dropdown when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    new SearchDropdown();
});
