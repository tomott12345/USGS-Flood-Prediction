package main

import "testing"

func TestValidateSiteCode(t *testing.T) {
	cases := []struct {
		in    string
		valid bool
	}{
		{"01388500", true},
		{"01553990", true},
		{"123456789012345", true},   // 15 digits, upper bound
		{"1234567", false},          // 7 digits, too short
		{"1234567890123456", false}, // 16 digits, too long
		{"", false},
		{"0138850x", false},
		{"01388500; rm -rf /", false},
	}
	for _, c := range cases {
		err := validateSiteCode(c.in)
		if c.valid && err != nil {
			t.Errorf("validateSiteCode(%q): expected valid, got error %v", c.in, err)
		}
		if !c.valid && err == nil {
			t.Errorf("validateSiteCode(%q): expected error, got none", c.in)
		}
	}
}

func TestValidateHorizons(t *testing.T) {
	cases := []struct {
		name  string
		in    []int
		valid bool
	}{
		{"default scope", []int{1, 3, 6}, true},
		{"single", []int{48}, true},
		{"empty", []int{}, false},
		{"zero", []int{0}, false},
		{"negative", []int{-1}, false},
		{"too large", []int{169}, false},
		{"duplicate", []int{1, 1}, false},
		{"too many", []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, false},
	}
	for _, c := range cases {
		err := validateHorizons(c.in)
		if c.valid && err != nil {
			t.Errorf("%s: expected valid, got error %v", c.name, err)
		}
		if !c.valid && err == nil {
			t.Errorf("%s: expected error, got none", c.name)
		}
	}
}

func TestValidateDays(t *testing.T) {
	cases := []struct {
		in    int
		valid bool
	}{
		{60, true},
		{14, true},
		{120, true},
		{13, false},
		{121, false},
		{0, false},
		{-5, false},
	}
	for _, c := range cases {
		err := validateDays(c.in)
		if c.valid && err != nil {
			t.Errorf("validateDays(%d): expected valid, got error %v", c.in, err)
		}
		if !c.valid && err == nil {
			t.Errorf("validateDays(%d): expected error, got none", c.in)
		}
	}
}
